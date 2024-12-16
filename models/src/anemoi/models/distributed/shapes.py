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


def get_shape_shards(tensor: Tensor, dim: int, model_comm_group: Optional[ProcessGroup] = None) -> list:
    """Get shape of tensor shards."""
    assert dim < tensor.dim(), f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}"

    comm_size = 1 if not model_comm_group else dist.get_world_size(group=model_comm_group)
    return [list(x.shape) for x in torch.tensor_split(tensor, comm_size, dim=dim)]


def change_channels_in_shape(shape_list: list, channels: int) -> list:
    """Change the number of channels in the tensor shape definition list."""
    return [x[:-1] + [channels] for x in shape_list] if shape_list else []
