import numpy as np
import plotly.express as px
from PIL import Image

from EX4.ex4 import ex4


def load_image(path: str):
    return np.asarray(Image.open(path).convert('L'))


def show_array(array: np.ndarray, title: str):
    fig = px.imshow(array, color_continuous_scale='gray')
    fig.update_layout(title_text=title)
    fig.show()


if __name__ == '__main__':
    img = load_image('Desktop.png')
    crop_size = (31, 61)
    crop_center = (300, 200)
    image, crop_array, target_array = ex4(img, crop_size, crop_center, transpose=True)

    show_array(image, f'crop_center={crop_center}, crop_size={crop_size}')
    show_array(crop_array, f'crop_center={crop_center}, crop_size={crop_size}')
    show_array(target_array, f'crop_center={crop_center}, crop_size={crop_size}')
