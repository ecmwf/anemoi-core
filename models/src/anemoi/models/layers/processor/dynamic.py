# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Optional

import torch
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import change_channels_in_shape
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.chunk import GraphTransformerProcessorChunk
from anemoi.models.layers.graph import TrainableTensor
from anemoi.models.layers.processor.base import BaseProcessor
from anemoi.utils.config import DotDict


class DynamicGraphTransformerProcessor(BaseProcessor):
    """Processor."""

    def __init__(
        self,
        num_layers: int,
        layer_kernels: DotDict,
        trainable_size: int = 8,
        num_channels: int = 128,
        num_chunks: int = 2,
        num_heads: int = 16,
        mlp_hidden_ratio: int = 4,
        cpu_offload: bool = False,
        subgraph_edge_index_name: str = "edge_index",
        subgraph_edge_attributes: Optional[list] = [],
        edge_dim: int = 0,
        **kwargs,
    ) -> None:
        """Initialize DynamicGraphTransformerProcessor.

        Parameters
        ----------
        num_layers : int
            Number of layers
        num_channels : int
            Number of channels
        num_chunks : int, optional
            Number of num_chunks, by default 2
        heads: int
            Number of heads to use, default 16
        mlp_hidden_ratio: int
            ratio of mlp hidden dimension to embedding dimension, default 4
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        """
        super().__init__(
            num_channels=num_channels,
            num_layers=num_layers,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            layer_kernels=layer_kernels,
        )
        self.edge_dim = edge_dim
        self.edge_attribute_names = subgraph_edge_attributes
        self.edge_index_name = subgraph_edge_index_name

        self.trainable = TrainableTensor(trainable_size=trainable_size, tensor_size=self.edge_dim)

        self.build_layers(
            GraphTransformerProcessorChunk,
            num_channels=num_channels,
            num_layers=self.chunk_size,
            layer_kernels=self.layer_factory,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            edge_dim=self.edge_dim,
        )

        self.offload_layers(cpu_offload)

    def forward(
        self,
        x: Tensor,
        subgraph: HeteroData,
        batch_size: int,
        shard_shapes: tuple[tuple[int], tuple[int]],
        model_comm_group: Optional[ProcessGroup] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        shape_nodes = change_channels_in_shape(shard_shapes, self.num_channels)
        edge_index = subgraph[self.edge_index_name].to(torch.int64)
        edge_attr = torch.cat([subgraph[attr] for attr in self.edge_attribute_names], axis=1)

        shapes_edge_attr = get_shard_shapes(edge_attr, 0, model_comm_group)
        edge_attr = shard_tensor(edge_attr, 0, shapes_edge_attr, model_comm_group)

        x, edge_attr = self.run_layers(
            (x, edge_attr),
            edge_index,
            (shape_nodes, shape_nodes, shapes_edge_attr),
            batch_size,
            model_comm_group,
        )

        return x
