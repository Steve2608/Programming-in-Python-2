import torch
from torch import nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm


def _eval(loader: DataLoader, model: nn.Module, optimizer=None):
    is_train = optimizer is not None
    device = next(model.parameters()).device

    mse_loss = MSELoss()
    total_loss = 0
    for _input, targets in tqdm(loader, total=len(loader)):
        _input = _input.to(device=device)

        compute_grad = torch.enable_grad() if is_train else torch.no_grad()
        with compute_grad:
            output, bool_masks = model(_input)
            predictions = [output[i, 0, bool_masks[i]] for i in range(len(output))]
            losses = torch.stack(
                [mse_loss(prediction, target.to(device=device).view(-1)) for
                 prediction, target in zip(predictions, targets)])
            loss = losses.sum()

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.detach().cpu().item()
    return total_loss / len(loader)


def train_eval(train_loader: DataLoader, model: nn.Module, optimizer):
    assert optimizer is not None
    model.train()
    return _eval(train_loader, model, optimizer)


def val_eval(val_loader: DataLoader, model: nn.Module):
    model.eval()
    return _eval(val_loader, model)


def test_eval(test_loader: DataLoader, model: nn.Module):
    model.eval()
    return _eval(test_loader, model)
