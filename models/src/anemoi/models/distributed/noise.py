# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import Optional

import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import ShardSizes


def draw_noise(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    model_comm_group: Optional[ProcessGroup] = None,
    shard_sizes: ShardSizes = None,
    shard_dim: int = -2,
) -> torch.Tensor:
    """Draw standard Gaussian noise of ``shape``, this rank's part of it when the nodes are split across ranks.

    Ranks of a model group share their random seed. Each rank draws the noise for all nodes of the group and
    keeps its own ``shard_sizes`` part along ``shard_dim``, so every node gets its own value, the ranks use up
    the same amount of randomness, and the result does not depend on the number of ranks.
    """
    if model_comm_group is None or not shard_sizes:
        return torch.randn(shape, device=device, dtype=dtype)

    ndim = len(shape)
    if not -ndim <= shard_dim < ndim:
        msg = f"Cannot shard random tensor of rank {ndim} along dimension {shard_dim}."
        raise ValueError(msg)

    shard_dim = shard_dim % ndim
    full_shape = list(shape)
    full_shape[shard_dim] = sum(shard_sizes)
    noise = torch.randn(tuple(full_shape), device=device, dtype=dtype)
    return shard_tensor(noise, shard_dim, list(shard_sizes), model_comm_group)
