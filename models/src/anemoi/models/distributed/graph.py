# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import torch
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.primitives import _alltoall_transpose
from anemoi.models.distributed.primitives import _expand_sharded_tensor
from anemoi.models.distributed.primitives import _gather
from anemoi.models.distributed.primitives import _halo_exchange
from anemoi.models.distributed.primitives import _halo_exchange_bwd
from anemoi.models.distributed.primitives import _reduce
from anemoi.models.distributed.primitives import _split
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.distributed.shapes import validate_dim
from anemoi.models.distributed.shapes import validate_shard_sizes
from anemoi.models.distributed.utils import model_is_distributed  # noqa: F401


def ensure_sharded(
    x: Tensor, dim: int, shard_sizes: ShardSizes, model_comm_group: ProcessGroup | None = None
) -> tuple[Tensor, ShardSizes]:
    """Ensure that the input tensor is sharded along the specified dimension.

    If ``shard_sizes`` is not None the tensor is assumed to already be
    sharded and a consistency check is performed.  Otherwise the tensor
    is sharded using balanced partitioning and the resulting sizes are
    returned.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dim : int
        Dimension along which to shard.
    shard_sizes : ShardSizes
        Per-rank partition sizes, or ``None`` if the tensor is replicated.
    model_comm_group : ProcessGroup, optional
        Model communication group.

    Returns
    -------
    tuple[Tensor, ShardSizes]
        The (possibly sharded) tensor and the shard sizes.
    """
    validate_dim(x, dim)
    if shard_sizes is not None:
        validate_shard_sizes(shard_sizes, model_comm_group)
        my_rank = model_comm_group.rank() if model_comm_group is not None else 0
        if shard_sizes[my_rank] != x.size(dim):
            raise ValueError(
                f"input tensor's size at dimension {dim} must match shard_sizes[{my_rank}] "
                f"({shard_sizes[my_rank]}), but got {x.size(dim)}"
            )
        return x, shard_sizes

    # x not sharded: get sizes and shard tensor accordingly
    shard_sizes = get_shard_sizes(x, dim, model_comm_group)
    return shard_tensor(x, dim, shard_sizes, model_comm_group), shard_sizes


def shard_tensor(
    input_: Tensor, dim: int, sizes: ShardSizes, mgroup: ProcessGroup | None, gather_in_backward: bool = True
) -> Tensor:
    """Shard tensor.

    Keeps only part of the tensor that is relevant for the current rank.

    Parameters
    ----------
    input_ : Tensor
        Input
    dim : int
        dimension along which to shard
    sizes : ShardSizes
        Per-rank shard sizes
    mgroup : ProcessGroup or None
        Model communication group. If ``None``, no communication is performed and this operation leaves the tensor unchanged.
    gather_in_backward : bool
        perform gather in backward, default True

    Returns
    -------
    Tensor
        Sharded tensor.
    """
    if mgroup is not None:
        validate_dim(input_, dim)
        validate_shard_sizes(sizes, mgroup)
        input_size = input_.size(dim)
        total_size = sum(sizes)
        if total_size != input_size:
            raise ValueError(
                f"sizes must sum exactly to {input_size} "
                f"(input tensor's size at dimension {dim}), but got {total_size}"
            )
    return _ShardParallelSection.apply(input_, dim, sizes, gather_in_backward, mgroup)


def gather_tensor(input_: Tensor, dim: int, sizes: ShardSizes, mgroup: ProcessGroup | None) -> Tensor:
    """Gather tensor.

    Gathers tensor shards from ranks.

    Parameters
    ----------
    input_ : Tensor
        Input
    dim : int
        dimension along which to gather
    sizes : ShardSizes
        Per-rank shard sizes
    mgroup : ProcessGroup or None
        Model communication group. If ``None``, no communication is performed and this operation leaves the tensor unchanged.

    Returns
    -------
    Tensor
        Gathered tensor.
    """
    if mgroup is not None:
        validate_dim(input_, dim)
        validate_shard_sizes(sizes, mgroup)
        rank = mgroup.rank()
        input_size = input_.size(dim)
        if input_size != sizes[rank]:
            raise ValueError(
                f"input tensor's size at dimension {dim} must match sizes[{rank}] "
                f"({sizes[rank]}), but got {input_size}"
            )
    return _GatherParallelSection.apply(input_, dim, sizes, mgroup)


def reduce_tensor(input_: Tensor, mgroup: ProcessGroup | None) -> Tensor:
    """Reduce tensor.

    Reduces tensor across ranks.

    Parameters
    ----------
    input_ : Tensor
        Input
    mgroup : ProcessGroup or None
        Model communication group. If ``None``, no communication is performed and this operation leaves the tensor unchanged.

    Returns
    -------
    Tensor
        Reduced tensor.
    """
    return _ReduceParallelSection.apply(input_, mgroup)


def sync_tensor(
    input_: Tensor,
    dim: int,
    sizes: ShardSizes,
    mgroup: ProcessGroup | None,
    gather_in_fwd: bool = True,
) -> Tensor:
    """Sync tensor.

    Perform a gather in the forward pass and an allreduce followed by a split in the backward pass.

    Parameters
    ----------
    input_ : Tensor
        Input
    dim : int
        dimension along which to gather
    sizes : ShardSizes
        Per-rank shard sizes
    mgroup : ProcessGroup or None
        Model communication group. If ``None``, no communication is performed and this operation leaves the tensor unchanged.

    Returns
    -------
    Tensor
        Synced tensor.
    """
    if mgroup is not None and gather_in_fwd and sizes is not None:
        validate_dim(input_, dim)
        validate_shard_sizes(sizes, mgroup)
        rank = mgroup.rank()
        input_size = input_.size(dim)
        if input_size != sizes[rank]:
            raise ValueError(
                f"input tensor's size at dimension {dim} must match sizes[{rank}] "
                f"({sizes[rank]}), but got {input_size}"
            )
    return _SyncParallelSection.apply(input_, dim, sizes, mgroup, gather_in_fwd)


def reduce_shard_tensor(input_: Tensor, dim: int, sizes: ShardSizes, mgroup: ProcessGroup | None) -> Tensor:
    """Reduces and then shards tensor.

    Perform an allreduce followed by a split in the forward pass and a gather in the backward pass.

    Parameters
    ----------
    input_ : Tensor
        Input
    dim : int
        dimension along which to gather
    sizes : ShardSizes
        Per-rank shard sizes
    mgroup : ProcessGroup or None
        Model communication group. If ``None``, no communication is performed and this operation leaves the tensor unchanged.

    Returns
    -------
    Tensor
        Reduced sharded tensor.
    """
    if mgroup is not None:
        validate_dim(input_, dim)
        validate_shard_sizes(sizes, mgroup)
        input_size = input_.size(dim)
        total_size = sum(sizes)
        if total_size != input_size:
            raise ValueError(
                f"sizes must sum exactly to {input_size} "
                f"(input tensor's size at dimension {dim}), but got {total_size}"
            )
    return _ReduceShardParallelSection.apply(input_, dim, sizes, mgroup)


def all_to_all_transpose(
    input_: Tensor,
    dim_split: int,
    split_sizes: ShardSizes,
    dim_concat: int,
    concat_sizes: ShardSizes,
    mgroup: ProcessGroup | None,
) -> Tensor:
    """All-to-all transpose.

    Switch the tensor from a dim_concat-sharded to a dim_split-sharded tensor via all-to-all transpose, reverse all-to-all in the backwards pass.

    Parameters
    ----------
    input_ : Tensor
        Input tensor to be transposed.
    dim_split : int
        Dimension along which to split the input tensor.
    split_sizes : ShardSizes
        Shapes of the split tensors.
    dim_concat : int
        Dimension along which to concatenate the transposed tensors.
    concat_sizes : ShardSizes
        Shapes of the concatenated tensors.
    mgroup : ProcessGroup or None
        Model communication group. If ``None``, no communication is performed and this operation leaves the tensor unchanged.

    Returns
    -------
    Tensor
        Transposed tensor.
    """
    if mgroup is not None:
        validate_dim(input_, dim_split)
        validate_dim(input_, dim_concat)
        ndim = input_.dim()
        if dim_split % ndim == dim_concat % ndim:
            raise ValueError(f"dim_split and dim_concat can not be the same, got {dim_split} and {dim_concat}")

        validate_shard_sizes(split_sizes, mgroup)
        validate_shard_sizes(concat_sizes, mgroup)
        split_size = input_.size(dim_split)
        total_split_size = sum(split_sizes)
        if total_split_size != split_size:
            raise ValueError(
                f"split_sizes must sum exactly to {split_size} "
                f"(input tensor's size at dimension {dim_split}), but got {total_split_size}"
            )
        rank = mgroup.rank()
        concat_size = input_.size(dim_concat)
        if concat_size != concat_sizes[rank]:
            raise ValueError(
                f"input tensor's size at dimension {dim_concat} must match concat_sizes[{rank}] "
                f"({concat_sizes[rank]}), but got {concat_size}"
            )

    return _AllToAllParallelSection.apply(input_, dim_split, split_sizes, dim_concat, concat_sizes, mgroup)


class _SyncParallelSection(torch.autograd.Function):
    """Sync the input from parallel section."""

    @staticmethod
    def forward(ctx, input_, dim_, sizes_, mgroup_, gather_in_fwd_=True):
        ctx.dim = dim_
        ctx.comm_group = mgroup_
        ctx.sizes = sizes_
        ctx.gather_in_fwd = gather_in_fwd_
        ctx.did_gather = bool(mgroup_ and gather_in_fwd_ and sizes_ is not None)
        if ctx.did_gather:
            return _gather(input_, dim_, sizes_, group=mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.comm_group:
            grad_output = _reduce(grad_output, group=ctx.comm_group)
            if ctx.did_gather:  # only split if we gathered in forward
                return (
                    _split(grad_output, ctx.dim, ctx.sizes, group=ctx.comm_group),
                    None,
                    None,
                    None,
                    None,
                )
        return grad_output, None, None, None, None


class _ReduceShardParallelSection(torch.autograd.Function):
    """All-reduce and shard the input from the parallel section."""

    # Modified from
    # Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

    @staticmethod
    def forward(ctx, input_, dim_, sizes_, mgroup_):
        ctx.dim = dim_
        ctx.comm_group = mgroup_
        ctx.sizes = sizes_
        if mgroup_:
            input_ = _reduce(input_, group=mgroup_)
            return _split(input_, dim_, sizes_, group=mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.comm_group:
            return (
                _gather(grad_output, ctx.dim, ctx.sizes, group=ctx.comm_group),
                None,
                None,
                None,
            )
        return grad_output, None, None, None


class _ShardParallelSection(torch.autograd.Function):
    """Split the input and keep only the relevant chunck to the rank."""

    # Modified from
    # Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

    @staticmethod
    def forward(ctx, input_, dim_, sizes_, gather_in_backward_, mgroup_):
        ctx.dim = dim_
        ctx.comm_group = mgroup_
        ctx.sizes = sizes_
        ctx.gather_in_backward = gather_in_backward_
        if mgroup_:
            return _split(input_, dim_, sizes_, group=mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.comm_group is not None:
            if ctx.gather_in_backward:
                return (
                    _gather(grad_output, ctx.dim, ctx.sizes, group=ctx.comm_group),
                    None,
                    None,
                    None,
                    None,
                )
            else:
                return (
                    _expand_sharded_tensor(grad_output, ctx.dim, ctx.sizes, group=ctx.comm_group),
                    None,
                    None,
                    None,
                    None,
                )
        return grad_output, None, None, None, None


class _GatherParallelSection(torch.autograd.Function):
    """Gather the input from parallel section and concatenate."""

    # Modified from
    # Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

    @staticmethod
    def forward(ctx, input_, dim_, sizes_, mgroup_):
        ctx.dim = dim_
        ctx.comm_group = mgroup_
        ctx.sizes = sizes_
        if mgroup_:
            return _gather(input_, dim_, sizes_, group=mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.comm_group:
            return (
                _split(grad_output, ctx.dim, ctx.sizes, group=ctx.comm_group),
                None,
                None,
                None,
            )
        return grad_output, None, None, None


class _ReduceParallelSection(torch.autograd.Function):
    """All-reduce the input from the parallel section."""

    @staticmethod
    def forward(ctx, input_, mgroup_):
        if mgroup_:
            return _reduce(input_, group=mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _AllToAllParallelSection(torch.autograd.Function):
    """All-to-all transpose along arbitrary split/concat dimensions.

    Forward: split along dim_split, all-to-all, concat along dim_concat.
    Backward: the inverse — split along dim_concat, all-to-all, concat along dim_split.
    """

    @staticmethod
    def forward(ctx, input_, dim_split_, split_sizes_, dim_concat_, concat_sizes_, mgroup_=None):
        ctx.dim_split = dim_split_
        ctx.split_sizes = split_sizes_
        ctx.dim_concat = dim_concat_
        ctx.concat_sizes = concat_sizes_
        ctx.comm_group = mgroup_
        if mgroup_:
            return _alltoall_transpose(input_, dim_split_, split_sizes_, dim_concat_, concat_sizes_, group=mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.comm_group:
            # Inverse: swap split/concat dims and sizes
            return (
                _alltoall_transpose(
                    grad_output,
                    ctx.dim_concat,
                    ctx.concat_sizes,
                    ctx.dim_split,
                    ctx.split_sizes,
                    group=ctx.comm_group,
                ),
                None,
                None,
                None,
                None,
                None,
            )
        return grad_output, None, None, None, None, None


def halo_exchange(input_: Tensor, halo_info, mgroup: ProcessGroup) -> Tensor:
    """Halo exchange for graph-partitioned node features.

    Sends inner node features to peer ranks that need them as halo
    nodes and receives halo node features from peers.  Returns the
    local features concatenated with received halo rows.

    Supports autograd: the backward pass reverses the communication and
    accumulates remote gradients at the ``send_indices`` positions.

    Parameters
    ----------
    input_ : Tensor
        Local node features, shape ``(num_local_nodes, ...)``.
    halo_info : HaloInfo
        Per-rank halo exchange metadata (from :func:`build_halo_info`).
    mgroup : ProcessGroup
        Model communication group.

    Returns
    -------
    Tensor
        ``(num_local_nodes + num_halo_nodes, ...)`` — local + halo features.
    """
    return _HaloExchangeParallelSection.apply(input_, halo_info, mgroup)


class _HaloExchangeParallelSection(torch.autograd.Function):
    """Halo exchange with autograd support.

    Forward: gather inner node features at ``send_indices``, exchange via
    all-to-all, concatenate received halo rows after local rows.

    Backward: reverse exchange — send halo gradients back to originating
    ranks, scatter-add received gradients at ``send_indices`` positions.
    """

    @staticmethod
    def forward(ctx, input_, halo_info_, mgroup_):
        ctx.send_indices = halo_info_.send_indices
        ctx.recv_counts = halo_info_.recv_counts
        ctx.num_local_nodes = halo_info_.num_local_nodes
        ctx.comm_group = mgroup_
        if mgroup_:
            return _halo_exchange(input_, halo_info_.send_indices, halo_info_.recv_counts, mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.comm_group:
            return (
                _halo_exchange_bwd(
                    grad_output,
                    ctx.send_indices,
                    ctx.recv_counts,
                    ctx.num_local_nodes,
                    ctx.comm_group,
                ),
                None,
                None,
            )
        return grad_output, None, None
