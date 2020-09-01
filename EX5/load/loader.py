from pathlib import Path
from random import randint
from typing import Tuple, Optional, List, Union, Dict, NamedTuple

import numpy as np
import torch
from PIL import Image
from torch import manual_seed
from torch.utils.data import Dataset, random_split
from torchvision.transforms import transforms as tfs
from torchvision.transforms.functional import resize

# ensure somewhat consistent executions
# np.random.seed(0)


class CroppedImage(NamedTuple):
    image_array: np.ndarray
    crop_array: np.ndarray
    target_array: np.ndarray


class SimpleImageDataset(Dataset):
    def __init__(self, root: Union[Path, str], transformer: Optional[tfs.Compose] = None,
                 cache: bool = False, uses_per_image: int = 1):
        super().__init__()

        self._transformer = transformer
        self._data = dict() if cache else None
        self._uses_per_image = uses_per_image

        self._paths = self._image_paths(Path(root))

    @property
    def has_cache(self) -> bool:
        return self._data is not None

    @property
    def transformer(self) -> Optional[tfs.Compose]:
        return self._transformer

    @property
    def paths(self) -> List[Path]:
        return list(self._paths.values())

    def __getitem__(self, index: int):
        if self.has_cache and index in self._data:
            return self._data[index]

        path = self._paths[index]
        image = Image.open(path)

        if self.transformer is not None:
            image = self.transformer(image)

        if self.has_cache:
            self._data[index] = image
        return image

    def __len__(self):
        return len(self._paths)

    def _image_paths(self, root: Union[Path, str]) -> Dict[int, Path]:
        paths = list(root.rglob('*.jpg')) * self._uses_per_image
        return {i: path for i, path in enumerate(paths)}


class CroppedImageDataset(Dataset):

    def __init__(self, dataset: Dataset, transformer: Optional[tfs.Compose] = None):
        self.dataset = dataset
        if transformer is not None:
            self.transformers = tfs.Compose([
                transformer, RandomResize(), RandomCrop(), SimpleNorm()
            ])
        else:
            self.transformers = tfs.Compose([RandomResize(), RandomCrop(), SimpleNorm()])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.transformers(self.dataset[index])


class RandomResize:

    def __init__(self,
                 # end exclusive
                 size_range: Tuple[int, int] = (70, 101)):
        self.size_range = size_range

    def __call__(self, image_pil: Image.Image):
        new_size = randint(self.size_range[0], self.size_range[1] - 1), \
                   randint(self.size_range[0], self.size_range[1] - 1)
        return resize(image_pil, new_size, interpolation=Image.LANCZOS)


class RandomCrop:
    def __init__(self,
                 # end exclusive
                 crop_range: Tuple[int, int] = (5, 22),
                 min_border: int = 20):
        self.crop_range = crop_range
        self.min_border = min_border

    def __call__(self, image_PIL: Image.Image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        size = image_PIL.size

        # generate odd crop numbers
        crop_x = randint(self.crop_range[0], self.crop_range[1] - 1)
        crop_y = randint(self.crop_range[0], self.crop_range[1] - 1)
        # make sure numbers are odd by setting LSB to 1
        x, y = (size[0] - self.min_border) | 1, (size[1] - self.min_border) | 1

        center_x = np.random.randint(self.min_border + crop_x // 2 + 1, x - crop_x // 2)
        center_y = np.random.randint(self.min_border + crop_y // 2 + 1, y - crop_y // 2)

        # image_PIL really is iterable
        # noinspection PyTypeChecker
        numpy_image = np.asarray(image_PIL, dtype=np.uint8).T

        return crop_image(numpy_image, crop_size=(crop_x, crop_y), crop_center=(center_x, center_y))


class SimpleNorm:
    def __call__(self, cropped: CroppedImage) -> CroppedImage:
        return CroppedImage(
            cropped.image_array.astype(np.float32) / 255,
            cropped.crop_array,
            cropped.target_array.astype(np.float32) / 255
        )


def custom_collate_fn(batch_list: List):
    max_x = 100  # max(map(lambda x: x[0].shape[0], batch_list))
    max_y = 100  # max(map(lambda y: y[0].shape[1], batch_list))
    data = torch.zeros((len(batch_list), 2, max_x, max_y), dtype=torch.float32)
    labels = [torch.tensor(item[2], dtype=torch.float32) for item in batch_list]
    for i, (image_array, crop_array, target_array) in enumerate(batch_list):
        data[i, 0, :image_array.shape[0], :image_array.shape[1]] = torch.from_numpy(image_array)
        data[i, 1, :image_array.shape[0], :image_array.shape[1]] = torch.from_numpy(crop_array)
    return data, labels


def crop_image(image_array: np.ndarray, crop_size: Tuple[int, int], crop_center: Tuple[int, int], *,
               copy: bool = True) -> CroppedImage:
    # no input checks necessary
    image = image_array.copy() if copy else image_array

    x, y = crop_center
    dx, dy = crop_size
    # integer division
    dx, dy = dx // 2, dy // 2

    # cropped out array
    target_array = image[x - dx:x + dx + 1, y - dy:y + dy + 1].copy()

    # bitmap array for which elements are cropped
    crop_array = np.zeros_like(image)
    crop_array[x - dx:x + dx + 1, y - dy:y + dy + 1] = 1

    # original array with cropped area set to 0
    image[x - dx:x + dx + 1, y - dy:y + dy + 1] = 0

    return CroppedImage(image, crop_array, target_array)


def train_test_split(dataset: Dataset, train: float = 0.8, val: float = 0.1, test: float = 0.1,
                     seed: int = 0):
    if not (0.999 <= train + val + test <= 1.001):
        raise ValueError(f'train ({train}), val ({val}) and test ({test}) '
                         f'do not sum up to 1: {train + val + test}!')

    n = len(dataset)
    n_train, n_val = int(n * train), int(n * val)
    n_test = n - (n_train + n_val)

    manual_seed(seed)

    return random_split(dataset, (n_train, n_val, n_test))


if __name__ == '__main__':
    ds = SimpleImageDataset('../data')
    print(len(ds))
    crop_ds = CroppedImageDataset(ds)
    print(len(crop_ds))
    for elem in crop_ds[1]:
        print(elem.shape)

    train, val, test = train_test_split(crop_ds)

    print(train, len(train))
