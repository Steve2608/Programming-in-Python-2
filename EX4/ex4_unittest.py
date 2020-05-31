import unittest

import numpy as np
from PIL import Image

from ex4 import ex4


class CropTest(unittest.TestCase):

    @staticmethod
    def test_keywords():
        ex4(image_array=np.random.random((100, 100)), crop_size=(1, 1), crop_center=(40, 40))

    def test_first_param_not_numpy_array(self):
        with self.assertRaises(ValueError):
            ex4([[1], [2]], (10, 10), (10, 10))

    def test_even_crop_size(self):
        with self.assertRaises(ValueError):
            ex4(np.asarray([1]), crop_size=(10, 10), crop_center=(20, 20))

        with self.assertRaises(ValueError):
            ex4(np.asarray([1]), crop_size=(11, 10), crop_center=(20, 20))

    def test_crop_center_distance(self):
        with self.assertRaises(ValueError):
            ex4(np.random.random((43, 43)), crop_size=(10, 10), crop_center=(20, 20))

    def test_specific_image(self):
        # load as grayscale
        desktop = np.asarray(Image.open('Desktop.png').convert('L'))

        # eh whatever
        actual_crop = np.asarray([[47, 46, 46, 45, 45, 45, 79],
                                  [50, 49, 48, 48, 47, 47, 46],
                                  [54, 53, 52, 51, 51, 50, 49],
                                  [59, 58, 57, 56, 55, 54, 53],
                                  [140, 104, 74, 61, 60, 59, 58],
                                  [158, 158, 158, 141, 106, 76, 64],
                                  [201, 170, 158, 158, 158, 158, 141],
                                  [255, 255, 237, 201, 170, 158, 158],
                                  [255, 255, 255, 255, 255, 237, 201],
                                  [255, 255, 255, 255, 255, 255, 255],
                                  [255, 255, 255, 255, 255, 255, 255]])

        crop_size = (11, 7)
        crop_center = (300, 220)
        image, crop_array, target_array = ex4(desktop, crop_size, crop_center)

        image_actual = desktop.copy()
        image_actual[295:306, 217:224] = 0
        self.assertEqual(0, np.sum(image_actual != image), 'Image did not match exactly')
        self.assertEqual(desktop.shape, image.shape, 'Shapes of image did not match')

        self.assertEqual(desktop.shape, crop_array.shape, 'Shape of crop_array did not match')
        bit_map_actual = np.zeros_like(desktop)
        bit_map_actual[295:306, 217:224] = 1
        self.assertEqual(0, np.sum(bit_map_actual != crop_array), 'Bitmap did not match exactly')

        self.assertEqual(crop_size, target_array.shape,
                         'Shape of target_array did not match exactly')
        self.assertEqual(0, np.sum(actual_crop != target_array),
                         'Target_array did not match exactly')


if __name__ == '__main__':
    unittest.main()
