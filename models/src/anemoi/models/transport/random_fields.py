# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0.

from __future__ import annotations

from typing import Optional

import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import ShardSizes


def randn_with_grid_sharding(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    model_comm_group: Optional[ProcessGroup] = None,
    grid_shard_sizes: ShardSizes = None,
    shard_dim: int = -2,
) -> torch.Tensor:
    """Create Gaussian noise once on the full grid, then keep this rank's shard.

    Model-parallel ranks normally share the same random seed. If each rank
    creates only its local grid shard, all ranks would receive identical noise.
    Creating the full field first and sharding it gives each grid point a
    distinct value while keeping ranks reproducible.
    """
    if model_comm_group is None or not grid_shard_sizes:
        return torch.randn(shape, device=device, dtype=dtype)

    ndim = len(shape)
    if not -ndim <= shard_dim < ndim:
        msg = f"Cannot shard random tensor of rank {ndim} along dimension {shard_dim}."
        raise ValueError(msg)

    shard_dim = shard_dim % ndim
    full_shape = list(shape)
    full_shape[shard_dim] = sum(grid_shard_sizes)
    noise = torch.randn(tuple(full_shape), device=device, dtype=dtype)
    return shard_tensor(noise, shard_dim, list(grid_shard_sizes), model_comm_group)


def randn_like_with_grid_sharding(
    tensor: torch.Tensor,
    *,
    model_comm_group: Optional[ProcessGroup] = None,
    grid_shard_sizes: ShardSizes = None,
    shard_dim: int = -2,
) -> torch.Tensor:
    """Create Gaussian noise like ``tensor``, with grid sharding."""
    return randn_with_grid_sharding(
        tuple(tensor.shape),
        device=tensor.device,
        dtype=tensor.dtype,
        model_comm_group=model_comm_group,
        grid_shard_sizes=grid_shard_sizes,
        shard_dim=shard_dim,
    )
