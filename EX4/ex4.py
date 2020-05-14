from typing import Tuple

import numpy as np


def ex4(image_array: np.ndarray, crop_size: Tuple[int, int], crop_center: Tuple[int, int],
        copy: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _check_input(crop_center, crop_size, image_array)

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

    return image, crop_array, target_array


def _check_input(crop_center, crop_size, image_array) -> None:
    if not isinstance(image_array, np.ndarray):
        raise ValueError(f'image_array is not a numpy array instance! (was {type(image_array)})')
    if len(crop_size) != 2:
        raise ValueError(f'Length of crop_size is != 2 ({len(crop_size)})')
    if len(crop_center) != 2:
        raise ValueError(f'Length of crop_center is != 2 ({len(crop_center)})')

    # checking for positivity of integers
    if any(d < 0 for d in crop_size):
        raise ValueError(f'Not all values in crop_size are positive: ({crop_size})')
    if any(d < 0 for d in crop_center):
        raise ValueError(f'Not all values in crop_center are positive: ({crop_center})')

    if any(d % 2 == 0 for d in crop_size):
        raise ValueError(f'Not all values in crop_size are odd: ({crop_size})')
    _check_border_distance(image_array, crop_size, crop_center, distance=20)


def _check_border_distance(image_array: np.ndarray, crop_size: Tuple[int, int],
                           crop_center: Tuple[int, int], distance: int) -> None:
    x, y = crop_center
    dx, dy = crop_size
    dx, dy = dx // 2, dy // 2
    dim_x, dim_y = image_array.shape
    if any(x < distance for x in [x - dx, dim_x - (x + dx), y - dy, dim_y - (y + dy)]):
        raise ValueError(
            f'Using crop_center and crop_size, the distance to the border is < {distance}!')
