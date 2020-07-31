from typing import Tuple

import torch


def ex6(logits: torch.Tensor, activation_function, threshold: torch.Tensor,
        targets: torch.Tensor) -> Tuple[float, float, float, float, float, float]:
    if not torch.is_floating_point(logits):
        raise TypeError(f'Type of logits was not float! {logits.dtype}')
    if not isinstance(threshold, torch.Tensor):
        raise TypeError(f'threshold is not a Tensor! {type(threshold)}')
    if targets.dtype is not torch.bool:
        raise TypeError(f'Type of targets is not bool! {targets.dtype}')

    if logits.ndim != 1:
        raise ValueError(f'logits must be one-dimensional! {logits.ndim}')
    if targets.ndim != 1:
        raise ValueError(f'targets must be one-dimensional! {logits.ndim}')
    if len(logits) != len(targets):
        raise ValueError(f'Length of logits and target does not match! {len(logits)} != '
                         f'{len(targets)}')
    if not (True in targets and False in targets):
        raise ValueError(
            'targets must contain at least one positive and negative class-label!')

    pred = torch.ge(activation_function(logits), threshold)

    tp = torch.sum(pred & targets).double()
    fp = torch.sum(pred & ~targets).double()
    tn = torch.sum(~pred & ~targets).double()
    fn = torch.sum(~pred & targets).double()

    tpr, fpr = tp / (tp + fn), fp / (fp + tn)
    tnr, fnr = tn / (tn + fp), fn / (fn + tp)
    acc = torch.sum(torch.eq(pred, targets)).double() / len(pred)
    b_acc = (tpr + tnr) / 2

    return tpr.item(), tnr.item(), fpr.item(), fnr.item(), acc.item(), b_acc.item()
