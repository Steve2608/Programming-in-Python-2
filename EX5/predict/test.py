import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Union, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from load.loader import SimpleNorm, crop_image
from models.simpleCNN import SimpleCNN
from predict.util import load_config, load_pkl, save_pkl


def _plot_predicted(image_arrays: Sequence[np.ndarray], predictions: Sequence[np.ndarray],
                    masks: Sequence[np.ndarray], writer: SummaryWriter):
    fig, ax = plt.subplots(3, 1)

    for i, (image, prediction, mask) in enumerate(zip(image_arrays, predictions, masks)):
        ax[0].clear()
        ax[0].set_title(f'Image_{i}')
        ax[0].imshow(image, cmap=plt.cm.gray, interpolation='none')
        ax[0].set_axis_off()

        ax[1].clear()
        ax[1].set_title(f'Prediction_{i}')
        ax[1].imshow(prediction, cmap=plt.cm.gray, interpolation='none')
        ax[1].set_axis_off()

        ax[2].clear()
        ax[2].set_title(f'Merged_{i}')
        merged = image.copy()
        mask = mask[0, :merged.shape[0], :merged.shape[1]]
        merged[mask] = prediction.flatten()
        ax[2].imshow(merged, cmap=plt.cm.gray, interpolation='none')
        ax[2].set_axis_off()

        fig.tight_layout()
        writer.add_figure('test/figures', fig, global_step=i)


def main(model_path: Union[str, Path], samples_path: Union[str, Path],
         config_path: Union[str, Path], pkl_path: Union[str, Path], tb_path: Union[str, Path]):
    config = load_config(config_path)

    tb = Path(tb_path)
    shutil.rmtree(tb, ignore_errors=True)
    tb.mkdir(exist_ok=True)

    network_spec, train_params = config['network_config'], config['train_params']

    # model
    model = SimpleCNN(**network_spec)
    model.to(device=config['device'])
    model.load_state_dict(torch.load(model_path))

    norm = SimpleNorm()
    images, crop_sizes, crop_centers = load_pkl(samples_path).values()

    predictions, masks = [], []
    model.eval()

    with torch.no_grad():
        # see custom_collate_fn
        max_x = 100  # max(map(lambda x: x[0].shape[0], images))
        max_y = 100  # max(map(lambda y: y[0].shape[1], images))
        for i, (image_array, crop_size, crop_center) in enumerate(
                zip(images, crop_sizes, crop_centers)):
            image_array, crop_array, crop_center = norm(
                crop_image(image_array, crop_size, crop_center))
            data = torch.zeros((1, 2, max_x, max_y), dtype=torch.float32)
            data[0, 0, :image_array.shape[0], :image_array.shape[1]] = torch.from_numpy(image_array)
            data[0, 1, :image_array.shape[0], :image_array.shape[1]] = torch.from_numpy(crop_array)
            data = data.cuda()
            output, bool_mask = model(data)
            prediction = output[0, 0, bool_mask[0]].reshape(crop_size)
            predictions.append((prediction.detach().cpu().numpy() * 256).astype(np.uint8))
            masks.append(bool_mask.detach().cpu().numpy())

    save_pkl(pkl_path, predictions)
    _plot_predicted(images, predictions, masks, SummaryWriter(log_dir=str(tb_path)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-model_path', type=str, required=True,
                        help='PyTorch Model to use')
    parser.add_argument('-sample_path', type=str, required=True,
                        help='pkl file to make predictions for')
    parser.add_argument('-config_path', type=str, required=True,
                        help='Configuration for CNN')
    parser.add_argument('-pkl_path', type=str, required=True,
                        help='Out path for pkl predictions')
    parser.add_argument('-tb_path', type=str, required=True,
                        help='Path for tensorboard')
    args = parser.parse_args()

    main(args.model_path, args.sample_path, args.config_path, args.pkl_path, args.tb_path)
