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


def _gather(
    input_: Tensor,
    dim_: int,
    shapes: tuple,
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
    tensor_list = [
        torch.empty(
            shapes[rank], dtype=input_.dtype, layout=input_.layout, device=input_.device, memory_format=input_format
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
