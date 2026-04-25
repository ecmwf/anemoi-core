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

from anemoi.models.distributed.shapes import ShardShapes
from anemoi.models.distributed.shapes import expand_shard_shapes
from anemoi.models.distributed.utils import get_memory_format


def _split(input_: Tensor, dim_: int, shapes_: ShardShapes, group: Optional[ProcessGroup] = None) -> Tensor:
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

    # Bypass the function if we are using only 1 GPU.
    comm_size = dist.get_world_size(group=group)
    if comm_size == 1:
        return input_

    # sanity checks
    assert dim_ < input_.dim(), f"Error, cannot split along {dim_} for tensor with {input_.dim()} dimensions."

    input_list = torch.split(input_, shapes_, dim=dim_)

    rank = dist.get_rank(group=group)
    output = input_list[rank].contiguous(memory_format=input_format)

    return output


def _gather(
    input_: Tensor,
    dim_: int,
    shapes: ShardShapes,
    gather_in_backward: bool = True,
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

    # get input format
    input_format = get_memory_format(input_)

    comm_size = dist.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if comm_size == 1:
        return input_

    # sanity checks
    assert dim_ < input_.dim(), f"Error, cannot gather along {dim_} for tensor with {input_.dim()} dimensions."

    # Size and dimension.
    comm_rank = dist.get_rank(group=group)

    input_ = input_.contiguous(memory_format=input_format)

    all_shards_equal_shape = all(shape == shapes[0] for shape in shapes)

    if dim_ == 0 and all_shards_equal_shape:  # requirement for all_gather_into_tensor
        out_shape = list(input_.shape)
        out_shape[dim_] = sum(shapes)

        output = torch.empty(
            out_shape, dtype=input_.dtype, layout=input_.layout, device=input_.device, memory_format=input_format
        )

        dist.all_gather_into_tensor(output, input_, group=group)
    else:
        tensor_shapes = expand_shard_shapes(input_, dim_, shapes)

        tensor_list = [
            torch.empty(
                tensor_shapes[rank],
                dtype=input_.dtype,
                layout=input_.layout,
                device=input_.device,
                memory_format=input_format,
            )
            for rank in range(comm_size)
        ]

        tensor_list[comm_rank] = input_
        if gather_in_backward:
            dist.all_gather(tensor_list, input_, group=group)

        # Note: torch.cat already creates a contiguous tensor.
        output = torch.cat(tensor_list, dim=dim_).contiguous(memory_format=input_format)

    return output


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

    comm_size = dist.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
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
    split_sizes : ShardShapes
        Size of each chunk along `dim_split`, one per rank
    dim_concat : int
        Dimension along which to concatenate the received chunks (can be negative)
    concat_sizes : ShardShapes
        Size of each received chunk along `dim_concat`, one per rank
    group : ProcessGroup, optional
        Process group

    Returns
    -------
    Tensor
        Result of the all-to-all exchange
    """
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
