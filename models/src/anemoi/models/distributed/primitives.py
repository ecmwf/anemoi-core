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

from anemoi.models.distributed.utils import get_memory_format


def _split(input_: Tensor, dim_: int, shapes_: tuple, group: Optional[ProcessGroup] = None) -> Tensor:
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

    input_list = torch.split(input_, [x[dim_] for x in shapes_], dim=dim_)

    rank = dist.get_rank(group=group)
    output = input_list[rank].contiguous(memory_format=input_format)

    return output


def _gather_into_tensor(
    input_: Tensor,
    dim_: int,
    shapes: tuple,
    group: ProcessGroup,
) -> Tensor:
    input_format = get_memory_format(input_)
    input_ = input_.contiguous(memory_format=input_format)

    out_shape = list(input_.shape)
    out_shape[dim_] = sum(shape[dim_] for shape in shapes)

    output = torch.empty(
        out_shape, dtype=input_.dtype, layout=input_.layout, device=input_.device, memory_format=input_format
    )

    dist.all_gather_into_tensor(output, input_, group=group)

    return output


def _gather_with_padding(
    input_: Tensor,
    dim_: int,
    shapes: tuple,
    group: ProcessGroup,
) -> Tensor:
    input_format = get_memory_format(input_)
    input_ = input_.contiguous(memory_format=input_format)
    dim = dim_ % input_.dim()

    max_shape_dim = max(shape[dim] for shape in shapes)
    padded_shape = list(input_.shape)
    padded_shape[dim] = max_shape_dim

    tensor_list = [
        torch.empty(
            padded_shape, dtype=input_.dtype, layout=input_.layout, device=input_.device, memory_format=input_format
        )
        for _ in range(len(shapes))
    ]

    # pad input_ to match max size in dim_
    pad_size = max_shape_dim - input_.shape[dim]
    if pad_size > 0:
        # pad format: (left_pad, right_pad, left_pad, right_pad, ...) descending from last dim
        pad = (0, 0) * (input_.dim() - dim - 1) + (0, pad_size)
        input_ = torch.nn.functional.pad(input_, pad, mode="constant", value=0)

    dist.all_gather(tensor_list, input_, group=group)

    # remove padding
    tensor_list = [torch.narrow(t, dim, 0, shape[dim]) for t, shape in zip(tensor_list, shapes)]

    return torch.cat(tensor_list, dim=dim).contiguous(memory_format=input_format)


def _gather_default(
    input_: Tensor,
    dim_: int,
    shapes: tuple,
    group: ProcessGroup,
) -> Tensor:
    """Gather using all_gather with pre-allocated buffers."""
    input_format = get_memory_format(input_)
    input_ = input_.contiguous(memory_format=input_format)

    tensor_list = [
        torch.empty(
            shapes[rank],
            dtype=input_.dtype,
            layout=input_.layout,
            device=input_.device,
            memory_format=input_format,
        )
        for rank in range(len(shapes))
    ]

    dist.all_gather(tensor_list, input_, group=group)

    return torch.cat(tensor_list, dim=dim_).contiguous(memory_format=input_format)


def _gather(
    input_: Tensor,
    dim_: int,
    shapes: tuple,
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

    # Bypass the function if we are using only 1 GPU.
    if dist.get_world_size(group=group) == 1:
        return input_

    # sanity checks
    assert (
        -input_.dim() <= dim_ < input_.dim()
    ), f"Error, cannot gather along {dim_} for tensor with {input_.dim()} dimensions."

    all_shards_equal_shape = all(shape == shapes[0] for shape in shapes)
    if dim_ == 0 and all_shards_equal_shape:  # requirement for all_gather_into_tensor
        return _gather_into_tensor(input_, dim_, shapes, group)

    requires_pad = dist.get_backend(group) == "gloo" and not all_shards_equal_shape
    if requires_pad:
        return _gather_with_padding(input_, dim_, shapes, group)

    return _gather_default(input_, dim_, shapes, group)


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


def _gather_channels_alltoall(input_: Tensor, shapes: list, group: Optional[ProcessGroup] = None) -> Tensor:
    """Apply all_to_all to go from channel-parallel to sequence-parallel.

    Split input along sequence dimension and join after all_to_all along channel dimension.
    Input: each GPU has full sequence, different channel parts
    Output: each GPU has different sequence parts, all channels
    """
    comm_size = dist.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if comm_size == 1:
        return input_
    myrank = dist.get_rank(group=group)

    # get input format
    input_format = get_memory_format(input_)

    # Split along sequence dimension
    input_list = [x.contiguous() for x in torch.tensor_split(input_, comm_size, dim=-2)]

    # Get total channels from shapes (original shape before channel splitting)
    channels = [x.shape[-1] for x in torch.tensor_split(torch.empty(shapes[myrank][-1], device="meta"), comm_size)]
    seq_per_rank = [x.shape[-2] for x in input_list]

    output_list = [
        torch.empty(
            (*input_list[0].shape[:-2], seq_per_rank[myrank], channels[rank]),
            dtype=input_.dtype,
            layout=input_.layout,
            device=input_.device,
            memory_format=input_format,
        )
        for rank in range(comm_size)
    ]
    dist.all_to_all(output_list, input_list, group=group)

    return torch.cat(output_list, dim=-1).contiguous(memory_format=input_format)


def _split_channels_alltoall(input_: Tensor, shapes: list, group: Optional[ProcessGroup] = None) -> Tensor:
    """Apply all_to_all along the head dimension.

    Split input along dimension dim_split and join after all_to_all along last dimesion.
    """
    comm_size = dist.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if comm_size == 1:
        return input_
    myrank = dist.get_rank(group=group)

    # get input format
    input_format = get_memory_format(input_)

    input_list = [x.contiguous() for x in torch.tensor_split(input_, comm_size, dim=-1)]

    input_shape = [x.shape for x in input_list]  # (... n c)
    channels_per_rank = [x.shape[-1] for x in input_list]

    output_list = [
        torch.empty(
            (*input_shape[rank][:-2], shapes[rank][-2], channels_per_rank[myrank]),
            dtype=input_.dtype,
            layout=input_.layout,
            device=input_.device,
            memory_format=input_format,
        )
        for rank in range(comm_size)
    ]

    dist.all_to_all(output_list, input_list, group=group)

    return torch.cat(output_list, dim=-2).contiguous(memory_format=input_format)
