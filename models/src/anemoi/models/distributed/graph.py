# (C) Copyright 2024 Anemoi contributors.
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
from anemoi.models.distributed.primitives import _gather
from anemoi.models.distributed.primitives import _reduce
from anemoi.models.distributed.primitives import _split
from anemoi.models.distributed.shapes import ShardShapes
from anemoi.models.distributed.shapes import get_shard_shapes


def ensure_sharded(
    x: Tensor, dim: int, shard_shapes: ShardShapes, model_comm_group: ProcessGroup | None = None
) -> tuple[Tensor, ShardShapes]:
    """Ensure that the input tensor is sharded along the specified dimension.

    If ``shard_shapes`` is not None the tensor is assumed to already be
    sharded and a consistency check is performed.  Otherwise the tensor
    is sharded using balanced partitioning and the resulting shapes are
    returned.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dim : int
        Dimension along which to shard.
    shard_shapes : ShardShapes
        Per-rank partition sizes, or ``None`` if the tensor is replicated.
    model_comm_group : ProcessGroup, optional
        Model communication group.

    Returns
    -------
    tuple[Tensor, ShardShapes]
        The (possibly sharded) tensor and the shard shapes.
    """
    if shard_shapes is not None:
        my_rank = model_comm_group.rank() if model_comm_group is not None else 0
        assert shard_shapes[my_rank] == x.shape[dim], (
            f"Error, expected shard shape {shard_shapes[my_rank]} along dimension {dim} "
            f"for rank {my_rank}, but got {x.shape[dim]}"
        )
        return x, shard_shapes

    # x not sharded: get shapes and shard tensor accordingly
    shard_shapes = get_shard_shapes(x, dim, model_comm_group)
    return shard_tensor(x, dim, shard_shapes, model_comm_group), shard_shapes


def shard_tensor(
    input_: Tensor, dim: int, shapes: ShardShapes, mgroup: ProcessGroup, gather_in_backward: bool = True
) -> Tensor:
    """Shard tensor.

    Keeps only part of the tensor that is relevant for the current rank.

    Parameters
    ----------
    input_ : Tensor
        Input
    dim : int
        dimension along which to shard
    shapes : ShardShapes
        Shapes of sharded Tensors
    mgroup : ProcessGroup
        model communication group
    gather_in_backward : bool
        perform gather in backward, default True

    Returns
    -------
    Tensor
        Sharded tensor.
    """
    return _ShardParallelSection.apply(input_, dim, shapes, gather_in_backward, mgroup)


def gather_tensor(input_: Tensor, dim: int, shapes: ShardShapes, mgroup: ProcessGroup) -> Tensor:
    """Gather tensor.

    Gathers tensor shards from ranks.

    Parameters
    ----------
    input_ : Tensor
        Input
    dim : int
        dimension along which to gather
    shapes : ShardShapes
        Shapes of sharded Tensors
    mgroup : ProcessGroup
        model communication group

    Returns
    -------
    Tensor
        Gathered tensor.
    """
    return _GatherParallelSection.apply(input_, dim, shapes, mgroup)


def reduce_tensor(input_: Tensor, mgroup: ProcessGroup) -> Tensor:
    """Reduce tensor.

    Reduces tensor across ranks.

    Parameters
    ----------
    input_ : Tensor
        Input
    mgroup : ProcessGroup
        model communication group

    Returns
    -------
    Tensor
        Reduced tensor.
    """
    return _ReduceParallelSection.apply(input_, mgroup)


def sync_tensor(input_: Tensor, dim: int, shapes: tuple, mgroup: ProcessGroup, gather_in_fwd: bool = True) -> Tensor:
    """Sync tensor.

    Perform a gather in the forward pass and an allreduce followed by a split in the backward pass.

    Parameters
    ----------
    input_ : Tensor
        Input
    dim : int
        dimension along which to gather
    shapes : ShardShapes
        Shapes of sharded Tensors
    mgroup : ProcessGroup
        model communication group

    Returns
    -------
    Tensor
        Synced tensor.
    """
    return _SyncParallelSection.apply(input_, dim, shapes, mgroup, gather_in_fwd)


def reduce_shard_tensor(input_: Tensor, dim: int, shapes: ShardShapes, mgroup: ProcessGroup) -> Tensor:
    """Reduces and then shards tensor.

    Perform an allreduce followed by a split in the forward pass and a gather in the backward pass.

    Parameters
    ----------
    input_ : Tensor
        Input
    dim : int
        dimension along which to gather
    shapes : ShardShapes
        Shapes of sharded Tensors
    mgroup : ProcessGroup
        model communication group

    Returns
    -------
    Tensor
        Reduced sharded tensor.
    """
    return _ReduceShardParallelSection.apply(input_, dim, shapes, mgroup)


def all_to_all_transpose(
    input_: Tensor,
    dim_split: int,
    split_sizes: ShardShapes,
    dim_concat: int,
    concat_sizes: ShardShapes,
    mgroup: ProcessGroup,
) -> Tensor:
    """All-to-all transpose.

    Switch the tensor from a dim_concat-sharded to a dim_split-sharded tensor via all-to-all transpose, reverse all-to-all in the backwards pass.

    Parameters
    ----------
    input_ : Tensor
        Input tensor to be transposed.
    dim_split : int
        Dimension along which to split the input tensor.
    split_sizes : ShardShapes
        Shapes of the split tensors.
    dim_concat : int
        Dimension along which to concatenate the transposed tensors.
    concat_sizes : ShardShapes
        Shapes of the concatenated tensors.
    mgroup : ProcessGroup
        Model communication group.

    Returns
    -------
    Tensor
        Transposed tensor.
    """
    return _AllToAllParallelSection.apply(input_, dim_split, split_sizes, dim_concat, concat_sizes, mgroup)


class _SyncParallelSection(torch.autograd.Function):
    """Sync the input from parallel section."""

    @staticmethod
    def forward(ctx, input_, dim_, shapes_, mgroup_, gather_in_fwd_=True):
        ctx.dim = dim_
        ctx.comm_group = mgroup_
        ctx.shapes = shapes_
        ctx.gather_in_fwd = gather_in_fwd_
        if mgroup_ and gather_in_fwd_:
            return _gather(input_, dim_, shapes_, group=mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.comm_group:
            grad_output = _reduce(grad_output, group=ctx.comm_group)
            if ctx.gather_in_fwd:  # only split if we gathered in forward
                return (
                    _split(grad_output, ctx.dim, ctx.shapes, group=ctx.comm_group),
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
    def forward(ctx, input_, dim_, shapes_, mgroup_):
        ctx.dim = dim_
        ctx.comm_group = mgroup_
        ctx.shapes = shapes_
        if mgroup_:
            input_ = _reduce(input_, group=mgroup_)
            return _split(input_, dim_, shapes_, group=mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.comm_group:
            return (
                _gather(grad_output, ctx.dim, ctx.shapes, group=ctx.comm_group),
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
    def forward(ctx, input_, dim_, shapes_, gather_in_backward_, mgroup_):
        ctx.dim = dim_
        ctx.comm_group = mgroup_
        ctx.shapes = shapes_
        ctx.gather_in_backward = gather_in_backward_
        if mgroup_:
            return _split(input_, dim_, shapes_, group=mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.comm_group:
            return (
                _gather(
                    grad_output, ctx.dim, ctx.shapes, gather_in_backward=ctx.gather_in_backward, group=ctx.comm_group
                ),
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
    def forward(ctx, input_, dim_, shapes_, mgroup_):
        ctx.dim = dim_
        ctx.comm_group = mgroup_
        ctx.shapes = shapes_
        if mgroup_:
            return _gather(input_, dim_, shapes_, group=mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.comm_group:
            return (
                _split(grad_output, ctx.dim, ctx.shapes, group=ctx.comm_group),
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
