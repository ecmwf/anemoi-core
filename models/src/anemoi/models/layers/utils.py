# (C) Copyright 2024-2026 Anemoi contributors.
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

from hydra.errors import InstantiationException
from hydra.utils import instantiate
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.utils.config import DotDict

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
    """Load layer kernels from the config.

    This function tries to load the layer kernels from the config. If the layer kernel is not supplied, it will fall back to the torch.nn implementation.

    Parameters
    ----------
    kernel_config : DotDict
        Kernel configuration, e.g. {"Linear": {"_target_": "torch.nn.Linear"}}
    instance : bool
        If True, instantiate the kernels. If False, return the config.
        This is useful for testing purposes.
        Defaults to True.

    Returns
    -------
    DotDict
        Container with layer factories.
    """
    # If self.layer_kernels entry is missing from the config, use torch.nn kernels
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

    # Loop through all kernels in the layer_kernels config entry and try import them
    for name, kernel_entry in {**default_kernels, **kernel_config}.items():
        if instance:
            try:
                layer_kernels[name] = instantiate(kernel_entry, _partial_=True)
            except InstantiationException:
                LOGGER.info(
                    f"{kernel_entry['_target_']} not found! Check your config.model.layer_kernel. {name} entry. Maybe your desired kernel is not installed or the import string is incorrect?"
                )
                raise InstantiationException
            else:
                LOGGER.info(f"{name} kernel: {kernel_entry['_target_']}.")
        else:
            layer_kernels[name] = kernel_entry
    return layer_kernels

# Every dropout layer torch ships (all subclasses of the private _DropoutNd).
_DROPOUT_MODULES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)


def _uses_dropout(module: nn.Module) -> bool:
    """Whether ``module`` drops anything: a dropout layer, or a ``dropout_p`` (attention), above zero."""
    for submodule in module.modules():
        if isinstance(submodule, _DROPOUT_MODULES) and submodule.p > 0:
            return True
        dropout_p = getattr(submodule, "dropout_p", None)
        if isinstance(dropout_p, int | float) and dropout_p > 0:
            return True
    return False


def refuse_dropout_when_sharded(module: nn.Module, model_comm_group: ProcessGroup | None) -> None:
    """Refuse dropout when the model is sharded across GPUs.

    Ranks in a model group share a random seed, so dropout on sharded data would draw
    correlated masks rather than independent ones. Every processor and mapper
    ``forward`` calls this first (a test checks that each one does). It counts dropout
    layers and any ``dropout_p`` above zero, including dropout added after
    construction, such as a LoRA adapter's. Only training is refused: dropout does
    nothing in eval mode, so sharded inference keeps working.

    Parameters
    ----------
    module : nn.Module
        The processor or mapper about to run.
    model_comm_group : ProcessGroup, optional
        The model communication group; ``None`` or size 1 means not sharded.

    Raises
    ------
    ValueError
        If the module is training, the model is sharded, and the module uses dropout.
    """
    # The group is only asked for its size when there is dropout to refuse.
    if not module.training or model_comm_group is None or not _uses_dropout(module):
        return
    if model_comm_group.size() > 1:
        raise ValueError(f"Dropout is not supported when the model is sharded across {model_comm_group.size()} GPUs")
