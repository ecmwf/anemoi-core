# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Shallow local branch at output resolution for the diffusion downscaler.

Why this exists (fine-scale epic, 2026-09-02). The downscaler graph has edges from the
data grid to the hidden mesh, from the hidden mesh to itself, and from the hidden mesh
back to the data grid; there is no edge from one output point to another. Every output
point is therefore computed from its own feature vector and messages from three hidden
nodes, and the only lateral pathway below the hidden-mesh spacing is the lossy one-hop
route through a hidden node. This module adds a shallow graph-transformer operating
directly on the output grid over nearest-neighbour edges, so that adjacent output points
can shape the fine band together.

Design rules:
* built only when configured; when absent nothing is constructed and no random numbers
  are drawn, so the model without the option is bit-identical to pristine;
* the output head is initialised to zero, so a freshly built branch is an exact no-op and a
  warm start from a checkpoint without the branch reproduces that checkpoint exactly;
* the branch output is added to the decoder output BEFORE the EDM preconditioning combines it
  with the skip term, i.e. ``D = c_skip * y_noised + c_out * (decoder + branch)``;
* it reuses ``GraphTransformerProcessor`` unchanged, including its sharding and halo code,
  on the ``data`` node set with a ``(data, to, data)`` edge set added to the graph file.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.processor import GraphTransformerProcessor
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)

# keys of the ``hres_branch`` config block that are forwarded to the constructor
CONFIG_KEYS = (
    "num_channels",
    "num_layers",
    "num_heads",
    "mlp_hidden_ratio",
    "num_chunks",
    "qk_norm",
    "shard_strategy",
    "graph_attention_backend",
    "detach_inputs",
    "zero_init_head",
    "layer_kernels",
)


def default_layer_kernels(num_channels: int, cond_dim: int) -> DotDict:
    """The lane's kernel block, re-sized to the branch width.

    The lane configuration hard-codes ``normalized_shape: 512`` in its ConditionalLayerNorm,
    which is why the branch cannot simply inherit the processor's ``layer_kernels``.
    """
    return DotDict(
        {
            "LayerNorm": {
                "_target_": "anemoi.models.layers.normalization.ConditionalLayerNorm",
                "normalized_shape": num_channels,
                "condition_shape": cond_dim,
                "zero_init": True,
                "autocast": False,
            },
            "Linear": {"_target_": "torch.nn.Linear"},
            "Activation": {"_target_": "torch.nn.GELU"},
            "QueryNorm": {"_target_": "anemoi.models.layers.normalization.AutocastLayerNorm", "bias": False},
            "KeyNorm": {"_target_": "anemoi.models.layers.normalization.AutocastLayerNorm", "bias": False},
        }
    )


class LocalHresBranch(nn.Module):
    """Local graph-transformer refinement on the output grid.

    Parameters
    ----------
    in_features : int
        Width of the raw per-point feature vector the model assembles (interpolated driver,
        high-resolution forcings, noised target, coordinates) PLUS the decoder output width.
    out_features : int
        Number of output channels (the decoder's output width).
    edge_dim : int
        Edge feature dimension of the ``(data, to, data)`` graph provider.
    num_channels, num_layers, num_heads, mlp_hidden_ratio, num_chunks, qk_norm
        Graph-transformer hyper-parameters. ``num_chunks`` groups layers for activation
        checkpointing (it must divide ``num_layers``); it is NOT edge chunking.
    cond_dim : int
        Width of the noise-conditioning vector (the model's ``noise_cond_dim``).
    layer_kernels : DotDict, optional
        Kernel block; defaults to the lane's block re-sized to ``num_channels``.
    shard_strategy : str
        ``edges`` (halo exchange over the sharded output grid) or ``heads``.
    detach_inputs : bool
        Detach the decoder output and the conditioning before the branch, so that a frozen
        trunk really receives no gradient through the branch.
    zero_init_head : bool
        Zero-initialise the output head (exact no-op at construction).
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        edge_dim: int,
        num_channels: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_hidden_ratio: int = 4,
        num_chunks: int = 4,
        qk_norm: bool = True,
        cond_dim: int = 16,
        layer_kernels: Optional[DotDict] = None,
        shard_strategy: str = "edges",
        graph_attention_backend: str = "triton",
        detach_inputs: bool = False,
        zero_init_head: bool = True,
    ) -> None:
        super().__init__()
        assert num_layers % num_chunks == 0, "hres_branch: num_layers must be divisible by num_chunks"
        self.detach_inputs = bool(detach_inputs)
        self.num_channels = int(num_channels)
        if layer_kernels is None:
            layer_kernels = default_layer_kernels(num_channels, cond_dim)

        self.proj = nn.Linear(in_features, num_channels)
        self.processor = GraphTransformerProcessor(
            num_layers=num_layers,
            num_channels=num_channels,
            num_chunks=num_chunks,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            edge_dim=edge_dim,
            qk_norm=qk_norm,
            cpu_offload=False,
            layer_kernels=layer_kernels,
            shard_strategy=shard_strategy,
            graph_attention_backend=graph_attention_backend,
        )
        self.out_norm = nn.LayerNorm(num_channels)
        self.head = nn.Linear(num_channels, out_features)
        if zero_init_head:
            nn.init.constant_(self.head.weight, 0.0)
            nn.init.constant_(self.head.bias, 0.0)
        LOGGER.info(
            "LocalHresBranch: in=%d out=%d width=%d layers=%d heads=%d chunks=%d edge_dim=%d "
            "shard_strategy=%s detach_inputs=%s zero_init_head=%s",
            in_features, out_features, num_channels, num_layers, num_heads, num_chunks, edge_dim,
            shard_strategy, self.detach_inputs, zero_init_head,
        )

    def forward(
        self,
        x_raw: torch.Tensor,
        x_dec: torch.Tensor,
        *,
        graph_provider,
        batch_size: int,
        node_shard_sizes,
        model_comm_group: Optional[ProcessGroup],
        cond: Optional[torch.Tensor],
        inputs_sharded: bool,
    ) -> torch.Tensor:
        """Return the branch correction on the same node layout as ``x_dec``.

        ``x_raw`` and ``x_dec`` are ``(batch*ensemble*grid, features)`` tensors. When the caller
        holds the full grid on every rank (``inputs_sharded`` False) but a model communication
        group exists, the nodes are split with the same balanced partition the conditioning
        tensor already uses, processed, and gathered back; when the caller already holds its
        shard, the branch runs on it directly.
        """
        if self.detach_inputs:
            x_dec = x_dec.detach()
            cond = cond.detach() if cond is not None else None
        h_in = torch.cat([x_raw, x_dec.to(x_raw.dtype)], dim=-1)

        gather_back = False
        if not inputs_sharded and model_comm_group is not None and model_comm_group.size() > 1:
            node_shard_sizes = get_shard_sizes(h_in, 0, model_comm_group)
            h_in = shard_tensor(h_in, 0, node_shard_sizes, model_comm_group)
            gather_back = True
        elif node_shard_sizes is None:
            node_shard_sizes = get_shard_sizes(h_in, 0, model_comm_group)

        edge_attr, edge_index, edge_shard_sizes = graph_provider.get_edges(
            batch_size=batch_size, model_comm_group=model_comm_group
        )
        shard_info = GraphShardInfo(nodes=node_shard_sizes, edges=edge_shard_sizes)

        h = self.proj(h_in)
        h = self.processor(
            x=h,
            batch_size=batch_size,
            shard_info=shard_info,
            edge_attr=edge_attr,
            edge_index=edge_index,
            model_comm_group=model_comm_group,
            cond=cond,
        )
        out = self.head(self.out_norm(h))
        if gather_back:
            out = gather_tensor(out, 0, node_shard_sizes, model_comm_group)
        return out.to(x_dec.dtype)
