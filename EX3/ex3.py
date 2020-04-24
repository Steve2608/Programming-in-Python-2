import argparse
from glob import iglob
from pathlib import Path
from typing import Tuple, Type, Iterator, List

import numpy as np
from PIL import Image
from tqdm import tqdm


class ImageNormalizer:

    def __init__(self, input_dir: str):
        if input_dir is None or len(input_dir) == 0:
            raise ValueError('Input directory was null or empty!')
        if not (path := Path(input_dir)).is_dir():
            raise ValueError(f"Input path was not a directory: '{input_dir}'")

        abs_paths = sorted(iglob(f'{str(path)}/*.jpg'))
        self._paths = [path for name in abs_paths if not (path := Path(name)).is_dir()]

    @property
    def file_names(self) -> List[str]:
        # disable writing access in list
        return [path.name for path in self._paths]

    @staticmethod
    def read_image_as_array(file_name: str, dtype: Type[np.float_]) -> np.ndarray:
        # convert np.unit8 to np.float64 explicitly; calculations should do that anyways
        # if there is an exception, just raise it
        return np.asarray(Image.open(file_name), dtype=dtype)

    def get_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        means, stds = [], []
        for file_name in tqdm(self._paths, desc='means/stds of files'):
            img = ImageNormalizer.read_image_as_array(file_name, np.float64)
            means.append(img.mean())
            stds.append(img.std())

        return np.asarray(means), np.asarray(stds)

    @property
    def stats(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_stats()

    def get_images(self) -> Iterator[np.ndarray]:
        for file_name in tqdm(self._paths, desc='Normalizing images'):
            img = ImageNormalizer.read_image_as_array(file_name, np.float32)
            # scaled = img / 255 changed in 04-04-2020 assignment
            # centered = scaled - scaled.mean()
            centered = img - img.mean()
            yield centered / centered.std()

    @property
    def images(self) -> Iterator[np.ndarray]:
        return self.get_images()

    # make class itself iterable as well; why not?
    __iter__ = get_images

    def __str__(self):
        return self.__class__.__name__ + f'({", ".join(self._paths)})'

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

    print('sum(means):', sum([imag.mean() for imag in imag_norm]))
    print('mean(vars):', np.mean([imag.var() for imag in imag_norm]))
