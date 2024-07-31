from typing import Optional

import torch
from torch import nn

from anemoi.training.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)


def grad_scaler(
    module: nn.Module,
    grad_in: tuple[torch.Tensor, ...],
    grad_out: tuple[torch.Tensor, ...],
) -> Optional[tuple[torch.Tensor, ...]]:
    """Scales the loss gradients.

    Uses the formula in https://arxiv.org/pdf/2306.06079.pdf, section 4.3.2

    Use <module>.register_full_backward_hook(grad_scaler, prepend=False) to register this hook.

    Parameters
    ----------
    module : nn.Module
        Loss object (not used)
    grad_in : tuple[torch.Tensor, ...]
        Loss gradients
    grad_out : tuple[torch.Tensor, ...]
        Output gradients (not used)

    Returns
    -------
    tuple[torch.Tensor, ...]
        Re-scaled input gradients
    """
    del module, grad_out  # not needed
    # loss = module(x_pred, x_true)
    # so - the first grad_input is that of the predicted state and the second is that of the "ground truth" (== zero)
    channels = grad_in[0].shape[-1]  # number of channels
    channel_weights = torch.reciprocal(torch.sum(torch.abs(grad_in[0]), dim=1, keepdim=True))  # channel-wise weights
    new_grad_in = (
        (channels * channel_weights) / torch.sum(channel_weights, dim=-1, keepdim=True) * grad_in[0]
    )  # rescaled gradient
    return new_grad_in, grad_in[1]
