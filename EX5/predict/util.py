import json
from pathlib import Path
from typing import Union

import torch
from torch.utils.tensorboard import SummaryWriter

from config import model_folder, best_model_name
from vis.plot import plot


def load_config(path: Union[Path, str]):
    with open(Path(path), 'r') as f:
        return json.load(f)


def store_best_model(epoch: int, model, result_root: Path, best_val_loss: float,
                     current_val_loss=None):
    torch.save(model.state_dict(), result_root / model_folder / f'model_{epoch}.pt')
    if current_val_loss is None:
        torch.save(model.state_dict(), result_root / model_folder / best_model_name)
        return best_val_loss
    if current_val_loss < best_val_loss:
        torch.save(model.state_dict(), result_root / model_folder / best_model_name)
        return current_val_loss
    return best_val_loss


def plot_samples(epoch: int, model: torch.nn.Module, sample_batch, sample_targets,
                 writer: SummaryWriter):
    model.eval()
    sample_batch = sample_batch.to('cuda:0')
    output, masks = model(sample_batch)
    sample_prediction = [output[i, 0, masks[i]] for i in range(len(output))]

    plot(
        sample_batch.cpu()[:, 0] * 255,
        sample_targets * 255,
        sample_prediction,
        writer,
        epoch
    )
