# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup


def get_shard_shapes(tensor: Tensor, dim: int, model_comm_group: Optional[ProcessGroup] = None) -> list[list[int]]:
    """Get shape of tensor shards split along a specific dimension."""
    assert dim < tensor.dim(), f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}"

    comm_size = 1 if not model_comm_group else dist.get_world_size(group=model_comm_group)
    return [list(x.shape) for x in torch.tensor_split(tensor, comm_size, dim=dim)]


def change_channels_in_shape(shape_list: list[list[int]], channels: int) -> list[list[int]]:
    """Change the number of channels in the tensor shape definition list."""
    return [x[:-1] + [channels] for x in shape_list] if shape_list else []


def apply_shard_shapes(tensor: Tensor, dim: int, shard_shapes_dim: list[int]) -> list[list[int]]:
    """Generalize shard shapes of a specific dimension to all dimensions of a given tensor."""
    assert dim < tensor.dim(), f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}"

    shard_shapes = [list(tensor.shape) for _ in range(len(shard_shapes_dim))]
    for i, shard_shape in enumerate(shard_shapes_dim):
        shard_shapes[i][dim] = shard_shape

    return shard_shapes


def get_or_apply_shard_shapes(
    x: Tensor, dim: int = 0, shard_shapes_dim: int = None, model_comm_group: Optional[ProcessGroup] = None
) -> list[list[int]]:
    if shard_shapes_dim is None:
        return get_shard_shapes(x, dim, model_comm_group)
    else:
        return apply_shard_shapes(x, dim, shard_shapes_dim)
