from pathlib import Path
from typing import Tuple, Type, Iterator, List, Union

import numpy as np
from PIL import Image
from tqdm import tqdm


class ImageNormalizer:

    def __init__(self, input_dir: str):
        if input_dir is None or len(input_dir) == 0:
            raise ValueError('Input directory was null or empty!')
        path = Path(input_dir)
        if not path.is_dir():
            raise ValueError(f"Input path was not a directory: '{input_dir}'")

        all_paths = path.glob('*.jpg')
        files_only = filter(lambda p: p.is_file(), all_paths)
        self._paths = sorted(files_only, key=str)

    @property
    def file_names(self) -> List[str]:
        # disable writing access in list
        return [path.name for path in self._paths]

    @staticmethod
    def read_image_as_array(path: Union[Path, str], dtype: Type[np.float_]) -> np.ndarray:
        return np.asarray(Image.open(path), dtype=dtype)

    def get_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        means, stds = [], []
        for path in tqdm(self._paths, desc='means/stds of files'):
            img = ImageNormalizer.read_image_as_array(path, np.float64)
            means.append(img.mean())
            stds.append(img.std())

        return np.asarray(means), np.asarray(stds)

    @property
    def stats(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_stats()

    def get_images(self) -> Iterator[np.ndarray]:
        for path in tqdm(self._paths, desc='Normalizing images'):
            img = ImageNormalizer.read_image_as_array(path, np.float32)
            centered = img - img.mean()
            yield centered / centered.std()

    @property
    def images(self) -> Iterator[np.ndarray]:
        return self.get_images()

    def __str__(self):
        return self.__class__.__name__ + ', '.join(map(str, self._paths))

    __repr__ = __str__
