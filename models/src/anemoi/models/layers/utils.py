# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import math
from typing import Optional

from torch import nn
from torch.utils.checkpoint import checkpoint

from anemoi.utils.config import DotDict
from anemoi.utils.parametrisation import ParametrisationError
from anemoi.utils.parametrisation import build

LOGGER = logging.getLogger(__name__)


def compute_mlp_hidden_dim(num_channels: int, mlp_hidden_ratio: float) -> int:
    """Compute integer hidden dimension from a (possibly fractional) MLP ratio.

    Parameters
    ----------
    num_channels : int
        Base channel width.
    mlp_hidden_ratio : float
        Multiplier used to derive hidden width.

    Returns
    -------
    int
        Rounded hidden width.
    """
    if not math.isfinite(mlp_hidden_ratio):
        msg = f"`mlp_hidden_ratio` must be finite, got {mlp_hidden_ratio}."
        raise ValueError(msg)
    if mlp_hidden_ratio <= 0:
        msg = f"`mlp_hidden_ratio` must be > 0, got {mlp_hidden_ratio}."
        raise ValueError(msg)

    # Round to nearest integer with halves rounding up.
    hidden_dim = int(num_channels * mlp_hidden_ratio + 0.5)
    if hidden_dim <= 0:
        msg = f"Computed hidden_dim must be > 0, got {hidden_dim}."
        raise ValueError(msg)
    return hidden_dim


class CheckpointWrapper(nn.Module):
    """Wrapper for checkpointing a module."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return checkpoint(self.module, *args, **kwargs, use_reentrant=False)


def maybe_checkpoint(func, enabled: bool, *args, **kwargs):
    """Conditionally apply gradient checkpointing to a function.

    Parameters
    ----------
    func : callable
        The function to potentially wrap with checkpointing
    enabled : bool
        Whether to apply gradient checkpointing
    *args, **kwargs
        Arguments to pass to the function

    Returns
    -------
    The result of calling func with the provided arguments
    """
    if enabled:
        return checkpoint(func, *args, **kwargs, use_reentrant=False)
    return func(*args, **kwargs)


def load_layer_kernels(kernel_config: Optional[DotDict] = None, instance: bool = True) -> DotDict["str" : nn.Module]:
    """Load layer kernels from the params.

    This function tries to load the layer kernels from the params. If the layer kernel is not supplied, it will fall back to the torch.nn implementation.

    Parameters
    ----------
    kernel_config : DotDict
        Kernel configuration, e.g. {"Linear": {"_target_": "torch.nn.Linear"}}
    instance : bool
        If True, instantiate the kernels. If False, return the params.
        This is useful for testing purposes.
        Defaults to True.

    Returns
    -------
    DotDict
        Container with layer factories.
    """
    # If self.layer_kernels entry is missing from the params, use torch.nn kernels
    default_kernels = {
        "Linear": {"_target_": "torch.nn.Linear"},
        "LayerNorm": {"_target_": "torch.nn.LayerNorm"},
        "Activation": {"_target_": "torch.nn.GELU"},
        "QueryNorm": {
            "_target_": "anemoi.models.layers.normalization.AutocastLayerNorm",
            "_partial_": True,
            "bias": False,
        },
        "KeyNorm": {
            "_target_": "anemoi.models.layers.normalization.AutocastLayerNorm",
            "_partial_": True,
            "bias": False,
        },
    }

    if kernel_config is None:
        kernel_config = DotDict()

    layer_kernels = DotDict()

    # Loop through all kernels in the layer_kernels params entry and try import them
    for name, kernel_entry in {**default_kernels, **kernel_config}.items():
        if instance:
            try:
                layer_kernels[name] = build(kernel_entry, _partial_=True)
            except ParametrisationError:
                LOGGER.info(
                    f"{kernel_entry['_target_']} not found! Check your params.model.layer_kernel. {name} entry. Maybe your desired kernel is not installed or the import string is incorrect?"
                )
                raise
            else:
                LOGGER.info(f"{name} kernel: {kernel_entry['_target_']}.")
        else:
            layer_kernels[name] = kernel_entry
    return layer_kernels
