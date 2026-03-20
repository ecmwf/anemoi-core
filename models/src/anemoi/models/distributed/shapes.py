# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from dataclasses import dataclass
from typing import Optional
from typing import Union

import torch.distributed as dist
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes

# Types for sharding metadata. These are per-rank partition sizes along one
# tensor dimension, not full per-rank tensor shapes.
ShardSizes = Union[list[int], None]
DatasetShardSizes = dict[str, ShardSizes]


@dataclass(frozen=True)
class GraphShardInfo:
    nodes: ShardSizes = None
    edges: ShardSizes = None

    def nodes_are_sharded(self):
        return self.nodes is not None

    def edges_are_sharded(self):
        return self.edges is not None


@dataclass(frozen=True)
class BipartiteGraphShardInfo:
    src_nodes: ShardSizes = None
    dst_nodes: ShardSizes = None
    edges: ShardSizes = None

    def src_is_sharded(self):
        return self.src_nodes is not None

    def dst_is_sharded(self):
        return self.dst_nodes is not None

    def edges_are_sharded(self):
        return self.edges is not None


def get_shard_sizes(tensor: Tensor, dim: int, model_comm_group: Optional[ProcessGroup] = None) -> ShardSizes:
    """Get per-rank shard sizes for a tensor split along a specific dimension."""
    assert dim < tensor.dim(), f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}"

    comm_size = 1 if not model_comm_group else dist.get_world_size(group=model_comm_group)
    return get_balanced_partition_sizes(tensor.shape[dim], comm_size)


def expand_shard_sizes_to_shapes(tensor: Tensor, dim: int, shard_sizes_dim: list[int]) -> list[list[int]]:
    """Expand per-dimension shard sizes to full per-rank tensor shapes."""
    assert dim < tensor.dim(), f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}"

    shard_shapes = [list(tensor.shape) for _ in range(len(shard_sizes_dim))]
    for i, shard_size in enumerate(shard_sizes_dim):
        shard_shapes[i][dim] = shard_size

    return shard_shapes
