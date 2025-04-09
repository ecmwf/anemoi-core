# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import Tensor

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.primitives import _gather
from anemoi.models.distributed.utils import get_memory_format

_SCALE_GRAD_DEFAULT = False


def gather_ensemble_members(
    input_: Tensor,
    dim: int,
    shapes: tuple,
    nens: int,
    ndevices: int,
    memspacing: int,
    mgroup: ProcessGroup,
    scale_gradients: bool = _SCALE_GRAD_DEFAULT,
) -> Tensor:
    """Gather ensemble members.

    Gather ensemble members and filter out duplicates due to model parallel runs.

    Parameters
    ----------
    input_ : Tensor
        Input
    dim : int
        dimension along which to gather
    shapes : tuple
        Shapes of sharded Tensors
        -> [input_.shape] * ens_comm_group_size
    nens : int
        number of unique ensemble members
        -> ens_comm_group_size * ensemble_size_per_device // model_comm_group_size
    ndevices : int,
        number of devices in ensemble group
    memspacing : int,
        spacing of unique members
        -> model_comm_group_size
    mgroup : ProcessGroup
        model communication group
    scale_gradients : bool
        scale gradients to compensate for splitting across workers

    Return
    -------
    Tensor
        Gathered ensemble members
    """
    return _GatherEnsembleMembers.apply(input_, dim, shapes, nens, ndevices, memspacing, scale_gradients, mgroup)


class _GatherEnsembleMembers(torch.autograd.Function):
    """Gather ensemble members and filter out duplicated members."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionContext,
        input_: Tensor,
        dim_: int,
        shapes_: tuple,
        nens_: int,
        ndevice_: int,
        memspacing_: int,
        scale_gradients_: bool,
        mgroup_: ProcessGroup,
    ) -> Tensor:
        ctx.dim = dim_
        ctx.comm_group = mgroup_
        ctx.shapes = shapes_
        ctx.nens = nens_
        ctx.ndevice = ndevice_
        ctx.memspacing = memspacing_
        ctx.scale_gradients = scale_gradients_
        if mgroup_:
            out = _gather(input_, dim_, shapes_, group=mgroup_)
            return _filter(out, dim_, ndevice_, memspacing_)

        return input_

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionContext,
        grad_output: Tensor,
    ) -> tuple[Tensor, None, None, None, None, None, None, None]:
        if ctx.comm_group:
            grad_output = _expand(grad_output, ctx.dim, ctx.nens, ctx.memspacing)
            grad_output = _split(grad_output, ctx.dim, ctx.shapes, group=ctx.comm_group)
            if ctx.scale_gradients:
                grad_output = grad_output * ctx.comm_group.size()
            return (
                grad_output,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        return grad_output, None, None, None, None, None, None, None


def _expand(input_: Tensor, dim_: int, nens_: int, memspacing_: int) -> Tensor:
    """Copy gradients of members to every (duplicated) member of original input."""
    if memspacing_ == 1:
        return input_

    assert input_.shape[dim_] % nens_ == 0

    member_index = [
        member.repeat(memspacing_) for member in torch.arange(input_.shape[dim_]).chunk(input_.shape[dim_] // nens_)
    ]
    member_index = torch.cat(member_index).tolist()

    # create output tensor with duplicated members
    input_list = torch.tensor_split(input_, input_.shape[dim_], dim=dim_)
    output = torch.cat([input_list[member] for member in member_index], dim=dim_).contiguous(
        memory_format=get_memory_format(input_),
    )
    return output * 1.0 / memspacing_  # to compensate for the gradient inflation from the member duplication


def _split(input_: Tensor, dim_: int, shapes_: tuple, group: ProcessGroup) -> Tensor:
    """Split the tensor along dim and keep the relevant slice."""
    # get input format
    input_format = get_memory_format(input_)

    # Bypass the function if we are using only 1 GPU.
    comm_size = dist.get_world_size(group=group)
    if comm_size == 1:
        return input_

    # sanity checks
    assert dim_ < input_.dim(), f"Error, cannot split along {dim_} for tensor with {input_.dim()} dimensions."

    input_list = torch.split(input_, [x[dim_] for x in shapes_], dim=dim_)

    rank = dist.get_rank(group=group)
    return input_list[rank].contiguous(memory_format=input_format)


def _filter(input_: Tensor, dim_: int, ndevices_: int, memspacing_: int) -> Tensor:
    """Only keep every x member input tensor along dim of members."""
    if memspacing_ == 1:
        return input_

    assert input_.shape[dim_] % ndevices_ == 0

    input_list = torch.chunk(input_, ndevices_, dim=dim_)[::memspacing_]
    return torch.cat(input_list, dim=dim_).contiguous(memory_format=get_memory_format(input_))
