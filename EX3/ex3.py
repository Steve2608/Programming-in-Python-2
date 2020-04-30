import argparse
from pathlib import Path
from typing import Tuple, Type, Iterator, List, Union

import numpy as np
from PIL import Image
from tqdm import tqdm


class ImageNormalizer:

    def __init__(self, input_dir: str):
        if input_dir is None or len(input_dir) == 0:
            raise ValueError('Input directory was null or empty!')
        if not (path := Path(input_dir)).is_dir():
            raise ValueError(f"Input path was not a directory: '{input_dir}'")

        all_paths = path.glob('*.jpg')
        files_only = filter(lambda p: p.is_file(), all_paths)
        self._paths = list(sorted(files_only))

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
        return self.__class__.__name__ + ", ".join(map(str, self._paths))

    __repr__ = __str__


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Programming in Python 2, Exercise 3')
    parser.add_argument('input_dir', type=str,
                        help='Relative or absolute path to input-directory. Path must point to a '
                             'directory. The directory must exist.')
    args = parser.parse_args()

    imag_norm = ImageNormalizer(args.input_dir)
    print(imag_norm)

    print('means/stds:', *imag_norm.stats, sep='\n', end='\n\n')

    print('sum(means):', sum([imag.mean() for imag in imag_norm.images]))
    print('mean(vars):', np.mean([imag.var() for imag in imag_norm.images]))
