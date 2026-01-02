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

def save_attention(epoch,class_name, run_id,alpha_attention, edge_index,layer=None):
    import numpy as np
    import os
    cwd = os.getcwd()
    os.makedirs(f"{cwd}/attention_weights/{run_id}", exist_ok=True)
    np.save(f"{cwd}/attention_weights/{run_id}/edge_index_{class_name}_{layer}.npy",edge_index.cpu().detach().numpy())
    np.save(f"{cwd}/attention_weights/{run_id}/attention_weights_{epoch}_{class_name}_{layer}.npy",alpha_attention.cpu().detach().numpy())

def node_level_entropy(edge_index, attn, num_nodes,index=1):
    """
    Vectorized head entropy.
    """
    import torch
    num_heads = attn.size(1)

    # Compute -α log α per edge
    edge_entropy = -(attn * torch.log(attn + 1e-12))  # [E, H]-    # Accumulate per node
    node_entropy = torch.zeros(num_nodes, num_heads, device=attn.device)  # [N, H]

    node_entropy.index_add_(0, edge_index[index], edge_entropy)

    # degree per node
    deg = torch.zeros(num_nodes, device=attn.device)
    deg.index_add_(0, edge_index[index],
                   torch.ones(edge_index.size(1), device=attn.device))

    # normalize per node
    node_entropy = node_entropy / torch.log(deg.clamp(min=2)).unsqueeze(1)

    # Average over nodes
    return node_entropy.mean(dim=0)


def save_attention(epoch, class_name, run_id, alpha_attention, edge_index, layer=None):
    import os

    import numpy as np

    cwd = os.getcwd()
    os.makedirs(f"{cwd}/attention_weights/{run_id}", exist_ok=True)
    if class_name == "processor":
        np.save(
            f"{cwd}/attention_weights/{run_id}/edge_index_{class_name}_{layer}.npy", edge_index.cpu().detach().numpy()
        )
        np.save(
            f"{cwd}/attention_weights/{run_id}/attention_weights_{epoch}_{class_name}_{layer}.npy",
            alpha_attention.cpu().detach().numpy(),
        )
    else:
        np.save(f"{cwd}/attention_weights/{run_id}/edge_index_{class_name}.npy", edge_index.cpu().detach().numpy())
        np.save(
            f"{cwd}/attention_weights/{run_id}/attention_weights_{epoch}_{class_name}.npy",
            alpha_attention.cpu().detach().numpy(),
        )


def node_level_entropy(edge_index, attn, num_nodes, index=1):
    """Vectorized head entropy."""
    import torch

    num_heads = attn.size(1)

    # Compute -α log α per edge
    edge_entropy = -(attn * torch.log(attn + 1e-12))  # [E, H]-    # Accumulate per node
    node_entropy = torch.zeros(num_nodes, num_heads, device=attn.device)  # [N, H]
    node_entropy.index_add_(0, edge_index[index], edge_entropy)

    # degree per node
    deg = torch.zeros(num_nodes, device=attn.device)
    deg.index_add_(0, edge_index[index], torch.ones(edge_index.size(1), device=attn.device))

    # normalize per node
    node_entropy = node_entropy / torch.log(deg.clamp(min=2)).unsqueeze(1)

    # Average over nodes
    return node_entropy.mean(dim=0)


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
