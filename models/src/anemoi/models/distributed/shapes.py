# (C) Copyright 2024-2026 Anemoi contributors.
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
ShardSizes = Union[list[int], tuple[int, ...], None]
DatasetShardSizes = dict[str, ShardSizes]


def validate_dim(tensor: Tensor, dim: int) -> None:
    """Check that a dimension is within valid range for a tensor.

    Parameters
    ----------
    tensor : Tensor
        Tensor whose dimensions define the valid range.
    dim : int
        Dimension to validate. Negative dimensions are supported.

    Raises
    ------
    TypeError
        If ``dim`` is not an integer.
    IndexError
        If ``dim`` is outside the valid range for ``tensor``.
    """
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise TypeError(f"Dimension must be an integer, but got {type(dim).__name__}")

    ndim = tensor.dim()
    if not -ndim <= dim < ndim:
        raise IndexError(f"Dimension out of range (expected to be in range of [-{ndim}, {ndim - 1}], but got {dim})")


def validate_shard_sizes(sizes: ShardSizes, mgroup: ProcessGroup) -> None:
    """Check that shard sizes are valid for the process group.

    Only call this for a group spanning more than one rank. Sharding metadata is
    unused where no communication happens, so a missing or single-rank group carries
    no notion of correct sizes and ``sizes`` is legitimately ``None`` there.

    Parameters
    ----------
    sizes : ShardSizes
        Shard sizes ordered by rank in the communication group.
    mgroup : ProcessGroup
        Communication group, spanning more than one rank.

    Raises
    ------
    TypeError
        If ``sizes`` is not a list or tuple of integers.
    ValueError
        If there is not one non-negative size per process.
    """
    comm_size = mgroup.size()
    assert comm_size > 1, f"Sharding metadata can't be verified for communication groups of size: {comm_size} < 2"

    if not isinstance(sizes, (list, tuple)):
        raise TypeError(f"Shard sizes must be a list or tuple of integers, but got {type(sizes).__name__}")
    if any(not isinstance(size, int) or isinstance(size, bool) for size in sizes):
        raise TypeError("Shard sizes must contain only integers")

    if len(sizes) != comm_size:
        raise ValueError(
            f"Shard sizes must contain one entry per process, but got {len(sizes)} entries "
            f"for a process group of size {comm_size}"
        )
    if any(size < 0 for size in sizes):
        raise ValueError(f"Shard sizes must contain only non-negative entries, but got {sizes}")


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
    validate_dim(tensor, dim)

    comm_size = 1 if not model_comm_group else dist.get_world_size(group=model_comm_group)
    return get_balanced_partition_sizes(tensor.shape[dim], comm_size)


def group_world_size(group: Optional[ProcessGroup] = None) -> int:
    """Number of ranks in ``group``, or 1 when no group is given.

    Uses ProcessGroup.size() rather than dist.get_world_size, so this is safe to
    call before (or without) init_process_group.
    """
    return group.size() if group is not None else 1


def check_shard_sizes_match_group(
    shard_sizes: ShardSizes,
    group: Optional[ProcessGroup] = None,
    *,
    context: str = "",
) -> None:
    """Validate that ``shard_sizes`` partitions a tensor over exactly ``group``.

    ``shard_sizes`` holds one entry per rank of the process group the tensor was sharded
    over, so sharding and (all)gathering must use the *same* group.

    ``None`` is accepted and returns without complaint: it means "replicated / not
    sharded", and whether replication is legal at a given point is the caller's decision,
    not this function's.

    Raises
    ------
    ValueError
        If ``shard_sizes`` does not have exactly one entry per rank of ``group``.
    """
    if shard_sizes is None:
        return

    world_size = group_world_size(group)
    if len(shard_sizes) != world_size:
        where = f" for {context}" if context else ""
        msg = (
            f"Shard/process-group mismatch{where}: shard_sizes has {len(shard_sizes)} "
            f"entries ({shard_sizes}) but the process group spans {world_size} rank(s). "
            "shard_sizes must have exactly one entry per rank of the group used to both "
            "shard and gather the tensor. In anemoi-training this almost always means a "
            "batch sharded over the *reader group* (dataloader.read_group_size ranks, see "
            "GriddedDataReader.set_reader_group_info) is being gathered over the *model "
            "communication group* (system.hardware.num_gpus_per_model ranks), or vice "
            "versa. Those two groups only coincide when "
            "dataloader.read_group_size == system.hardware.num_gpus_per_model."
        )
        raise ValueError(msg)


def expand_shard_sizes_to_shapes(
    tensor: Tensor, dim: int, shard_sizes_dim: list[int] | tuple[int, ...]
) -> list[list[int]]:
    """Expand per-dimension shard sizes to full per-rank tensor shapes."""
    validate_dim(tensor, dim)

    shard_shapes = [list(tensor.shape) for _ in range(len(shard_sizes_dim))]
    for i, shard_size in enumerate(shard_sizes_dim):
        shard_shapes[i][dim] = shard_size

    return shard_shapes
