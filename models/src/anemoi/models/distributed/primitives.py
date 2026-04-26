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

from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import expand_shard_sizes_to_shapes
from anemoi.models.distributed.utils import get_memory_format


def _split(input_: Tensor, dim_: int, sizes_: ShardSizes, group: Optional[ProcessGroup] = None) -> Tensor:
    """Split the tensor along dim and keep the relevant slice."""
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

    # get input format
    input_format = get_memory_format(input_)

    # Bypass the function if not in a distributed context or using only 1 GPU.
    if group is None or not dist.is_initialized():
        return input_
    comm_size = dist.get_world_size(group=group)
    if comm_size == 1:
        return input_

    # sanity checks
    assert dim_ < input_.dim(), f"Error, cannot split along {dim_} for tensor with {input_.dim()} dimensions."

    input_list = torch.split(input_, sizes_, dim=dim_)

    rank = dist.get_rank(group=group)
    output = input_list[rank].contiguous(memory_format=input_format)

    return output


def _gather_into_tensor(
    input_: Tensor,
    dim_: int,
    sizes: ShardSizes,
    group: ProcessGroup,
) -> Tensor:
    input_format = get_memory_format(input_)
    input_ = input_.contiguous(memory_format=input_format)

    out_shape = list(input_.shape)
    out_shape[dim_] = sum(sizes)  # expand shard sizes in dim_ to summed shape

    output = torch.empty(
        out_shape, dtype=input_.dtype, layout=input_.layout, device=input_.device, memory_format=input_format
    )

    dist.all_gather_into_tensor(output, input_, group=group)

    return output


def _gather_with_padding(
    input_: Tensor,
    dim_: int,
    sizes: ShardSizes,
    group: ProcessGroup,
) -> Tensor:
    input_format = get_memory_format(input_)
    input_ = input_.contiguous(memory_format=input_format)
    dim = dim_ % input_.dim()

    max_shape_dim = max(sizes)
    padded_shape = list(input_.shape)
    padded_shape[dim] = max_shape_dim

    tensor_list = [
        torch.empty(
            padded_shape, dtype=input_.dtype, layout=input_.layout, device=input_.device, memory_format=input_format
        )
        for _ in range(len(sizes))
    ]

    # pad input_ to match max size in dim_
    pad_size = max_shape_dim - input_.shape[dim]
    if pad_size > 0:
        # pad format: (left_pad, right_pad, left_pad, right_pad, ...) descending from last dim
        pad = (0, 0) * (input_.dim() - dim - 1) + (0, pad_size)
        input_ = torch.nn.functional.pad(input_, pad, mode="constant", value=0)

    dist.all_gather(tensor_list, input_, group=group)

    # remove padding
    tensor_list = [torch.narrow(t, dim, 0, size) for t, size in zip(tensor_list, sizes)]

    return torch.cat(tensor_list, dim=dim).contiguous(memory_format=input_format)


def _gather_default(
    input_: Tensor,
    dim_: int,
    sizes: ShardSizes,
    group: ProcessGroup,
) -> Tensor:
    """Gather using all_gather with pre-allocated buffers."""
    input_format = get_memory_format(input_)
    input_ = input_.contiguous(memory_format=input_format)
    full_shard_shapes = expand_shard_sizes_to_shapes(input_, dim_, sizes)

    tensor_list = [
        torch.empty(
            shape,
            dtype=input_.dtype,
            layout=input_.layout,
            device=input_.device,
            memory_format=input_format,
        )
        for shape in full_shard_shapes
    ]

    dist.all_gather(tensor_list, input_, group=group)

    return torch.cat(tensor_list, dim=dim_).contiguous(memory_format=input_format)


def _gather(
    input_: Tensor,
    dim_: int,
    sizes: ShardSizes,
    group: Optional[ProcessGroup] = None,
) -> Tensor:
    """Gather tensors and concatenate along the last dimension."""
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

    # Bypass the function if not in a distributed context or using only 1 GPU.
    if group is None or not dist.is_initialized():
        return input_
    if dist.get_world_size(group=group) == 1:
        return input_

    # sanity checks
    assert (
        -input_.dim() <= dim_ < input_.dim()
    ), f"Error, cannot gather along {dim_} for tensor with {input_.dim()} dimensions."

    all_shards_equal_shape = all(size == sizes[0] for size in sizes)
    if dim_ == 0 and all_shards_equal_shape:  # requirement for all_gather_into_tensor
        return _gather_into_tensor(input_, dim_, sizes, group)

    requires_pad = dist.get_backend(group) == "gloo" and not all_shards_equal_shape
    if requires_pad:
        return _gather_with_padding(input_, dim_, sizes, group)

    return _gather_default(input_, dim_, sizes, group)


def _reduce(input_: Tensor, use_fp32: bool = True, group: Optional[ProcessGroup] = None) -> Tensor:
    """All-reduce the input tensor across model parallel group."""
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

    # Bypass the function if not in a distributed context or using only 1 GPU.
    if group is None or not dist.is_initialized():
        return input_
    comm_size = dist.get_world_size(group=group)
    if comm_size == 1:
        return input_

    # All-reduce.
    if use_fp32:
        dtype = input_.dtype
        inputf_ = input_.float()
        dist.all_reduce(inputf_, group=group)
        input_ = inputf_.to(dtype)
    else:
        dist.all_reduce(input_, group=group)

    return input_


def _alltoallwrapper(output_list: list, input_list: list, group: ProcessGroup):
    """Wrapper function for all_to_all across NCCL, MPI and Gloo backends.
    There is no all_to_all primitive for the Gloo backend. In that case each
    process broadcasts its tensor asynchronously.

    Retuns nothing but modifies output_list in-place

    """
    comm_size = dist.get_world_size(group=group)

    if dist.get_backend(group) == "gloo":

        # Need to check torch version here bc the syntax for dist.send/recv changed in torch v2.6
        torch_version = torch.__version__.split(".")
        torch_major_version = int(torch_version[0])
        torch_minor_version = int(torch_version[1])
        if torch_major_version <= 2 and torch_minor_version < 6:
            raise NotImplementedError("Gloo all_to_all not implemented for torch < v2.6")

        reqs = []
        rank = dist.get_rank(group=group)
        # Here we implement the linear shift algorithm from Hofmann and Ruenger, 2013
        for i in range(0, comm_size):
            j = (i - rank + comm_size) % comm_size
            if j != rank:
                # exchange data with rank j
                reqs.append(dist.isend(input_list[j], group_dst=j, group=group))
                reqs.append(dist.irecv(output_list[j], group_src=j, group=group))
            else:
                output_list[rank] = input_list[rank]
        for req in reqs:
            req.wait()
    else:
        dist.all_to_all(output_list, input_list, group=group)


def _alltoall_transpose(
    input_: Tensor,
    dim_split: int,
    split_sizes: list[int],
    dim_concat: int,
    concat_sizes: list[int],
    group: Optional[ProcessGroup] = None,
) -> Tensor:
    """Unified all-to-all distributed transpose along arbitrary dimensions.

    Given a tensor that's distributed across ranks according to `concat_sizes`, reshard
    it to be distributed along `dim_split` using `split_sizes`. Done by splitting the tensor
    along `dim_split` using `split_sizes`, performing an all-to-all exchange, and concatenating
    the received tensors along `dim_concat`.

    Parameters
    ----------
    input_ : Tensor
        Input tensor
    dim_split : int
        Dimension along which to split the input (can be negative)
    split_sizes : ShardSizes
        Size of each chunk along `dim_split`, one per rank
    dim_concat : int
        Dimension along which to concatenate the received chunks (can be negative)
    concat_sizes : ShardSizes
        Size of each received chunk along `dim_concat`, one per rank
    group : ProcessGroup, optional
        Process group

    Returns
    -------
    Tensor
        Result of the all-to-all exchange
    """
    # Bypass the function if not in a distributed context or using only 1 GPU.
    if group is None or not dist.is_initialized():
        return input_
    comm_size = dist.get_world_size(group=group)
    if comm_size == 1:
        return input_

    myrank = dist.get_rank(group=group)
    input_format = get_memory_format(input_)

    # normalise negative dims
    ndim = input_.dim()
    dim_split = dim_split % ndim
    dim_concat = dim_concat % ndim

    # split input along dim_split
    input_list = [x.contiguous() for x in torch.split(input_, split_sizes, dim=dim_split)]

    # build output tensors: each has the shape of input_ but with
    # dim_split size = split_sizes[myrank] and dim_concat size = concat_sizes[rank]
    output_list = []
    for rank in range(comm_size):
        out_shape = list(input_.shape)
        out_shape[dim_split] = split_sizes[myrank]
        out_shape[dim_concat] = concat_sizes[rank]
        output_list.append(
            torch.empty(
                out_shape,
                dtype=input_.dtype,
                layout=input_.layout,
                device=input_.device,
                memory_format=input_format,
            )
        )

    _alltoallwrapper(output_list, input_list, group=group)

    return torch.cat(output_list, dim=dim_concat).contiguous(memory_format=input_format)


def _halo_exchange(
    x: Tensor,
    send_indices: tuple[Tensor, ...],
    recv_counts: tuple[int, ...],
    group: ProcessGroup,
) -> Tensor:
    """Forward halo exchange: gather inner node features and send to peers.

    For each peer rank *r*, gathers ``x[send_indices[r]]`` and sends it
    to rank *r*.  Receives ``recv_counts[r]`` rows from rank *r*.
    Returns *x* concatenated with the received halo rows, preserving the
    ordering expected by the relabeled local edge index.

    Parameters
    ----------
    x : Tensor
        Local node features, shape ``(num_local, ...)``.
    send_indices : tuple[Tensor, ...]
        Per-rank local indices of inner nodes to send.  Length = world size.
    recv_counts : tuple[int, ...]
        Per-rank number of halo rows to receive.  Length = world size.
    group : ProcessGroup
        Communication group.

    Returns
    -------
    Tensor
        ``(num_local + sum(recv_counts), ...)`` — local followed by halo rows.
    """
    comm_size = dist.get_world_size(group=group)
    if comm_size == 1:
        return x

    send_list = [x[idx].contiguous() for idx in send_indices]
    recv_list = [torch.empty((count, *x.shape[1:]), dtype=x.dtype, device=x.device) for count in recv_counts]

    _alltoallwrapper(recv_list, send_list, group=group)

    return torch.cat([x] + recv_list, dim=0)


def _halo_exchange_bwd(
    grad_output: Tensor,
    send_indices: tuple[Tensor, ...],
    recv_counts: tuple[int, ...],
    num_local_nodes: int,
    group: ProcessGroup,
) -> Tensor:
    """Backward of halo exchange.

    Splits *grad_output* into local and per-rank halo gradient portions.
    Sends halo gradients back to the ranks that originally owned those
    nodes and accumulates the received gradients at the ``send_indices``
    positions via scatter-add.

    Parameters
    ----------
    grad_output : Tensor
        Gradient w.r.t. the halo-exchange output, shape ``(total_nodes, ...)``.
    send_indices : tuple[Tensor, ...]
        Per-rank local indices (same as in the forward pass).
    recv_counts : tuple[int, ...]
        Per-rank halo counts (same as in the forward pass).
    num_local_nodes : int
        Number of local (inner) nodes.
    group : ProcessGroup
        Communication group.

    Returns
    -------
    Tensor
        Gradient w.r.t. the halo-exchange input, shape ``(num_local_nodes, ...)``.
    """
    comm_size = dist.get_world_size(group=group)
    if comm_size == 1:
        return grad_output

    grad_local = grad_output[:num_local_nodes].clone()

    # split halo portion into per-rank chunks (same ordering as forward recv)
    halo_grads = list(torch.split(grad_output[num_local_nodes:], list(recv_counts), dim=0))

    # reverse exchange: send halo grads to originating ranks,
    # receive grads for inner nodes we sent in the forward pass
    send_counts = [idx.size(0) for idx in send_indices]
    recv_list = [
        torch.empty((count, *grad_output.shape[1:]), dtype=grad_output.dtype, device=grad_output.device)
        for count in send_counts
    ]
    send_list = [g.contiguous() for g in halo_grads]

    _alltoallwrapper(recv_list, send_list, group=group)

    # scatter-add: accumulate received gradients into local grad
    all_recv = torch.cat(recv_list, dim=0)
    all_indices = torch.cat(list(send_indices), dim=0)
    if all_recv.numel() > 0:
        grad_local.index_add_(0, all_indices, all_recv)

    return grad_local
