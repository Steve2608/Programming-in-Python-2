import shutil
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Union, Tuple

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import src.eval.train_test_eval as tte
from src.load.loader import train_test_split, CroppedImageDataset, custom_collate_fn
from src.models.simpleCNN import SimpleCNN
from src.predict.util import load_config
from src.vis.plot import plot


def _plot_samples(epoch: int, model: torch.nn.Module, sample_batch, sample_targets,
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


def _setup_out_path(result_path: Union[str, Path]) -> Tuple[Path, Path, Path]:
    result_path = Path(result_path)
    result_path.mkdir(exist_ok=True)

    tb_path = result_path / 'tensorboard'
    shutil.rmtree(tb_path, ignore_errors=True)
    tb_path.mkdir(exist_ok=True)

    model_path = result_path / 'models'
    model_path.mkdir(exist_ok=True)

    return result_path, tb_path, model_path


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

    # dataset
    ds = CroppedImageDataset(dataset_path)

    # loaders
    train, val, test = train_test_split(ds, train=0.75, val=0.15, test=0.1)
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

        val_loss = tte.test_eval(val, model)
        print(f'val/loss: {val_loss}')
        writer.add_scalar(tag='val/loss', scalar_value=val_loss, global_step=epoch)

        if epoch % insight_interval == 0:
            _plot_samples(epoch, model, sample_input, sample_targets, writer)

    print('Best model evaluation...')
    test_loss = tte.test_eval(test, model)
    val_loss = tte.test_eval(val, model)

    print(test_loss, val_loss)
    print('Finished training process.')

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
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
