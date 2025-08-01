# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
from abc import ABC
from typing import Optional

from torch import Tensor
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper

from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import change_channels_in_shape
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)

# Number of chunks used in inference (https://github.com/ecmwf/anemoi-core/pull/406)
NUM_CHUNKS_INFERENCE = int(os.environ.get("ANEMOI_INFERENCE_NUM_CHUNKS", "1"))
NUM_CHUNKS_INFERENCE_MAPPER = int(os.environ.get("ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER", NUM_CHUNKS_INFERENCE))


class BaseMapper(nn.Module, ABC):
    """Base Mapper from souce dimension to destination dimension."""

    def __init__(
        self,
        *,
        in_channels_src: int,
        in_channels_dst: int,
        hidden_dim: int,
        out_channels_dst: Optional[int] = None,
        cpu_offload: bool = False,
        layer_kernels: DotDict,
        **kwargs,
    ) -> None:
        """Initialize BaseMapper."""
        super().__init__()

        self.in_channels_src = in_channels_src
        self.in_channels_dst = in_channels_dst
        self.hidden_dim = hidden_dim
        self.out_channels_dst = out_channels_dst
        self.layer_factory = load_layer_kernels(layer_kernels)
        self.activation = self.layer_factory.Activation()

        self.proc = NotImplemented

        self.offload_layers(cpu_offload)

    def offload_layers(self, cpu_offload):
        if cpu_offload:
            self.proc = nn.ModuleList([offload_wrapper(x) for x in self.proc])

    def pre_process(
        self, x, shard_shapes, model_comm_group=None, x_src_is_sharded=False, x_dst_is_sharded=False
    ) -> tuple[Tensor, Tensor, tuple[int], tuple[int]]:
        """Pre-processing for the Mappers.

        Splits the tuples into src and dst nodes and shapes as the base operation.

        Parameters
        ----------
        x : Tuple[Tensor]
            Data containing source and destination nodes and edges.
        shard_shapes : Tuple[Tuple[int], Tuple[int]]
            Shapes of the sharded source and destination nodes.
        model_comm_group : ProcessGroup
            Groups which GPUs work together on one model instance

        Return
        ------
        Tuple[Tensor, Tensor, Tuple[int], Tuple[int]]
            Source nodes, destination nodes, sharded source node shapes, sharded destination node shapes
        """
        shapes_src, shapes_dst = shard_shapes
        x_src, x_dst = x
        return x_src, x_dst, shapes_src, shapes_dst

    def post_process(self, x_dst, shapes_dst, model_comm_group=None, keep_x_dst_sharded=False) -> Tensor:
        """Post-processing for the mapper."""
        return x_dst


class BackwardMapperPostProcessMixin:
    """Post-processing for Backward Mapper from hidden -> data."""

    def post_process(self, x_dst, shapes_dst, model_comm_group=None, keep_x_dst_sharded=False):
        x_dst = self.node_data_extractor(x_dst)
        if not keep_x_dst_sharded:
            x_dst = gather_tensor(
                x_dst, 0, change_channels_in_shape(shapes_dst, self.out_channels_dst), model_comm_group
            )
        return x_dst


class ForwardMapperPreProcessMixin:
    """Pre-processing for Forward Mapper from data -> hidden."""

    def pre_process(self, x, shard_shapes, model_comm_group=None, x_src_is_sharded=False, x_dst_is_sharded=False):
        x_src, x_dst, shapes_src, shapes_dst = super().pre_process(
            x, shard_shapes, model_comm_group, x_src_is_sharded, x_dst_is_sharded
        )
        if not x_src_is_sharded:
            x_src = shard_tensor(x_src, 0, shapes_src, model_comm_group)
        if not x_dst_is_sharded:
            x_dst = shard_tensor(x_dst, 0, shapes_dst, model_comm_group)
        x_src = self.emb_nodes_src(x_src)
        x_dst = self.emb_nodes_dst(x_dst)
        shapes_src = change_channels_in_shape(shapes_src, self.hidden_dim)
        shapes_dst = change_channels_in_shape(shapes_dst, self.hidden_dim)
        return x_src, x_dst, shapes_src, shapes_dst
