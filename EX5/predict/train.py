import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Union, Tuple
from datetime import datetime

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import eval.train_test_eval as tte
from config import model_folder, tb_folder
from load.loader import SimpleImageDataset, train_test_split, CroppedImageDataset, custom_collate_fn
from predict.util import load_config, plot_samples
from models.simpleCNN import SimpleCNN


def _setup_out_path(result_path: Union[str, Path]) -> Tuple[Path, Path, Path]:
    result = Path(result_path)
    result.mkdir(exist_ok=True)

    tb_dir = result / tb_folder
    shutil.rmtree(tb_dir, ignore_errors=True)
    tb_dir.mkdir(exist_ok=True)

    model_dir = result / model_folder
    model_dir.mkdir(exist_ok=True)

    return result, tb_dir, model_dir


def main(dataset_path: Union[str, Path], config_path: Union[str, Path],
         result_path: Union[str, Path]):
    # paths and summary writer
    result_path, tb_path, model_path = _setup_out_path(result_path)
    writer = SummaryWriter(log_dir=str(tb_path))

    config = load_config(config_path)
    network_spec, train_params = config['network_config'], config['train_params']

    # params
    n_workers = config['n_workers']
    batch_size = config['batch_size']
    n_epochs = config['n_epochs']
    device = torch.device(config['device'])
    insight_interval = config['insight_interval']

    # loaders
    ds = SimpleImageDataset(dataset_path)
    train, val, test = train_test_split(ds, train=0.75, val=0.15, test=0.1)
    train, val, test = CroppedImageDataset(train), \
                       CroppedImageDataset(val), \
                       CroppedImageDataset(test)
    train = DataLoader(train, batch_size=batch_size, num_workers=n_workers,
                       collate_fn=custom_collate_fn)
    val = DataLoader(val, batch_size=batch_size, num_workers=n_workers,
                     collate_fn=custom_collate_fn)
    test = DataLoader(test, batch_size=batch_size, num_workers=n_workers,
                      collate_fn=custom_collate_fn)

    # model
    model = SimpleCNN(**network_spec)
    model.to(device=device)

    # optimizer
    optimizer = Adam(params=model.parameters(), **train_params)

    # tensorboard visuals
    sample_input, sample_targets = next(iter(val))

    print('Start Training: ')
    for epoch in range(n_epochs):
        print(f'Epoch: {epoch}')

        train_loss = tte.train_eval(train, model, optimizer)
        print(f'train/loss: {train_loss}')
        writer.add_scalar(tag='train/loss', scalar_value=train_loss, global_step=epoch)

        val_loss = tte.val_eval(val, model)
        print(f'val/loss: {val_loss}')
        writer.add_scalar(tag='val/loss', scalar_value=val_loss, global_step=epoch)

        if epoch % insight_interval == 0:
            plot_samples(epoch, model, sample_input, sample_targets, writer)

    print('Best model evaluation...')
    test_loss = tte.test_eval(test, model)
    val_loss = tte.val_eval(val, model)

    print(test_loss, val_loss)
    print('Finished training process.')

    timestamp = datetime.now().strftime("%Y%m%d%-%H%M%S")
    torch.save(model.state_dict(), model_path / f'model_{n_epochs}_{timestamp}.pt')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-dataset_path', type=str,
                        required=True,
                        help='Path to dataset')
    parser.add_argument('-config_path', type=str,
                        help='Path to config.json',
                        required=True)
    parser.add_argument('-result_path', type=str,
                        help='Path to results-folder',
                        required=True)
    args = parser.parse_args()

    main(args.dataset_path, args.config_path, args.result_path)
