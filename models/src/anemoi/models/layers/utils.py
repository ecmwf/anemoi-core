# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Optional

from hydra.errors import InstantiationException
from hydra.utils import instantiate
from torch import nn
from torch.utils.checkpoint import checkpoint

from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class CheckpointWrapper(nn.Module):
    """Wrapper for checkpointing a module."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return checkpoint(self.module, *args, **kwargs, use_reentrant=False)


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
