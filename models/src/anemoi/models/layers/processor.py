# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from abc import ABC
from typing import Optional

from torch import Tensor
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.khop_edges import sort_edges_1hop_sharding
from anemoi.models.distributed.shapes import change_channels_in_shape
from anemoi.models.distributed.shapes import get_shape_shards
from anemoi.models.layers.chunk import GNNProcessorChunk
from anemoi.models.layers.chunk import GraphTransformerProcessorChunk
from anemoi.models.layers.chunk import TransformerProcessorChunk
from anemoi.models.layers.graph import TrainableTensor
from anemoi.models.layers.mapper import GraphEdgeMixin


class BaseProcessor(nn.Module, ABC):
    """Base Processor."""

    def __init__(
        self,
        num_layers: int,
        *args,
        num_channels: int = 128,
        num_chunks: int = 2,
        activation: str = "GELU",
        cpu_offload: bool = False,
        positional_encoding_hidden: Optional[Tensor] = None,
        **kwargs,
    ) -> None:
        """Initialize BaseProcessor."""
        super().__init__()

        # Each Processor divides the layers into chunks that get assigned to each ProcessorChunk
        self.num_chunks = num_chunks
        self.num_channels = num_channels
        self.chunk_size = num_layers // num_chunks
        self.positional_encoding_hidden = positional_encoding_hidden

        assert (
            num_layers % num_chunks == 0
        ), f"Number of processor layers ({num_layers}) has to be divisible by the number of processor chunks ({num_chunks})."

    def offload_layers(self, cpu_offload):
        if cpu_offload:
            self.proc = nn.ModuleList([offload_wrapper(x) for x in self.proc])

    def build_layers(self, processor_chunk_class, *args, **kwargs) -> None:
        """Build Layers."""
        self.proc = nn.ModuleList(
            [
                processor_chunk_class(
                    *args,
                    **kwargs,
                )
                for _ in range(self.num_chunks)
            ],
        )

    def run_layers(self, data: tuple, *args, **kwargs) -> Tensor:
        """Run Layers with checkpoint."""
        for layer in self.proc:
            data = checkpoint(layer, *data, *args, **kwargs, use_reentrant=False)
        return data

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Example forward pass."""
        x = self.run_layers((x,), *args, **kwargs)
        return x


class TransformerProcessor(BaseProcessor):
    """Transformer Processor."""

    def __init__(
        self,
        num_layers: int,
        *args,
        window_size: Optional[int] = None,
        num_channels: int = 128,
        num_chunks: int = 2,
        activation: str = "GELU",
        cpu_offload: bool = False,
        positional_encoding_hidden: Optional[Tensor] = None,
        num_heads: int = 16,
        mlp_hidden_ratio: int = 4,
        dropout_p: float = 0.1,
        **kwargs,
    ) -> None:
        """Initialize TransformerProcessor.

        Parameters
        ----------
        num_layers : int
            Number of num_layers
        window_size: int,
            1/2 size of shifted window for attention computation
        num_channels : int
            number of channels
        heads: int
            Number of heads to use, default 16
        mlp_hidden_ratio: int
            ratio of mlp hidden dimension to embedding dimension, default 4
        activation : str, optional
            Activation function, by default "GELU"
        dropout_p: float, optional
            Dropout probability used for multi-head self attention, default 0.0
        """
        super().__init__(
            num_channels=num_channels,
            num_layers=num_layers,
            window_size=window_size,
            num_chunks=num_chunks,
            activation=activation,
            cpu_offload=cpu_offload,
            positional_encoding_hidden=positional_encoding_hidden,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
        )

        self.build_layers(
            TransformerProcessorChunk,
            num_channels=num_channels,
            mlp_hidden_ratio=mlp_hidden_ratio,
            num_heads=num_heads,
            num_layers=self.chunk_size,
            window_size=window_size,
            activation=activation,
            positional_encoding_hidden=positional_encoding_hidden,
            dropout_p=dropout_p,
        )

        self.offload_layers(cpu_offload)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_shapes: tuple[tuple[int], ...],
        model_comm_group: Optional[ProcessGroup] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        shape_nodes = change_channels_in_shape(shard_shapes, self.num_channels)
        if model_comm_group:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), "Only batch size of 1 is supported when model is sharded accross GPUs"

        (x,) = self.run_layers((x,), shape_nodes, batch_size, model_comm_group)

        return x


class GNNProcessor(GraphEdgeMixin, BaseProcessor):
    """GNN Processor."""

    def __init__(
        self,
        num_layers: int,
        *args,
        trainable_size: int = 8,
        num_channels: int = 128,
        num_chunks: int = 2,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        cpu_offload: bool = False,
        sub_graph: Optional[HeteroData] = None,
        sub_graph_edge_attributes: Optional[list[str]] = None,
        src_grid_size: int = 0,
        dst_grid_size: int = 0,
        **kwargs,
    ) -> None:
        """Initialize GNNProcessor.

        Parameters
        ----------
        num_channels : int
            Number of Channels
        num_layers : int
            Number of layers
        num_chunks : int, optional
            Number of num_chunks, by default 2
        mlp_extra_layers : int, optional
            Number of extra layers in MLP, by default 0
        activation : str, optional
            Activation function, by default "SiLU"
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        """
        super().__init__(
            num_channels=num_channels,
            num_layers=num_layers,
            num_chunks=num_chunks,
            activation=activation,
            cpu_offload=cpu_offload,
            mlp_extra_layers=mlp_extra_layers,
        )

        self._register_edges(sub_graph, sub_graph_edge_attributes, src_grid_size, dst_grid_size, trainable_size)

        self.trainable = TrainableTensor(trainable_size=trainable_size, tensor_size=self.edge_attr.shape[0])

        kwargs = {
            "num_layers": self.chunk_size,
            "mlp_extra_layers": mlp_extra_layers,
            "activation": activation,
            "edge_dim": None,
        }

        self.build_layers(GNNProcessorChunk, num_channels, **kwargs)

        kwargs["edge_dim"] = self.edge_dim  # Edge dim for first layer
        self.proc[0] = GNNProcessorChunk(num_channels, **kwargs)

        self.offload_layers(cpu_offload)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_shapes: tuple[tuple[int], tuple[int]],
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> Tensor:
        shape_nodes = change_channels_in_shape(shard_shapes, self.num_channels)
        edge_attr = self.trainable(self.edge_attr, batch_size)
        edge_index = self._expand_edges(self.edge_index_base, self.edge_inc, batch_size)
        target_nodes = sum(x[0] for x in shape_nodes)
        edge_attr, edge_index, shapes_edge_attr, shapes_edge_idx = sort_edges_1hop_sharding(
            target_nodes,
            edge_attr,
            edge_index,
            model_comm_group,
        )
        edge_index = shard_tensor(edge_index, 1, shapes_edge_idx, model_comm_group)
        edge_attr = shard_tensor(edge_attr, 0, shapes_edge_attr, model_comm_group)

        x, edge_attr = self.run_layers((x, edge_attr), edge_index, (shape_nodes, shape_nodes), model_comm_group)

        return x


class GraphTransformerProcessor(GraphEdgeMixin, BaseProcessor):
    """Processor."""

    def __init__(
        self,
        num_layers: int,
        trainable_size: int = 8,
        num_channels: int = 128,
        num_chunks: int = 2,
        num_heads: int = 16,
        mlp_hidden_ratio: int = 4,
        activation: str = "GELU",
        cpu_offload: bool = False,
        sub_graph: Optional[HeteroData] = None,
        sub_graph_edge_attributes: Optional[list[str]] = None,
        src_grid_size: int = 0,
        dst_grid_size: int = 0,
        **kwargs,
    ) -> None:
        """Initialize GraphTransformerProcessor.

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
        activation : str, optional
            Activation function, by default "GELU"
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        """
        super().__init__(
            num_layers=num_layers,
            num_channels=num_channels,
            num_chunks=num_chunks,
            activation=activation,
            cpu_offload=cpu_offload,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
        )

        self._register_edges(sub_graph, sub_graph_edge_attributes, src_grid_size, dst_grid_size, trainable_size)

        self.trainable = TrainableTensor(trainable_size=trainable_size, tensor_size=self.edge_attr.shape[0])

        self.build_layers(
            GraphTransformerProcessorChunk,
            num_channels=num_channels,
            num_layers=self.chunk_size,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            activation=activation,
            edge_dim=self.edge_dim,
        )

        self.offload_layers(cpu_offload)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_shapes: tuple[tuple[int], tuple[int]],
        model_comm_group: Optional[ProcessGroup] = None,
        *args,
        **kwargs,
    ) -> Tensor:

        shape_nodes = change_channels_in_shape(shard_shapes, self.num_channels)
        edge_attr = self.trainable(self.edge_attr, batch_size)

        edge_index = self._expand_edges(self.edge_index_base, self.edge_inc, batch_size)

        shapes_edge_attr = get_shape_shards(edge_attr, 0, model_comm_group)
        edge_attr = shard_tensor(edge_attr, 0, shapes_edge_attr, model_comm_group)

        x, edge_attr = self.run_layers(
            (x, edge_attr),
            edge_index,
            (shape_nodes, shape_nodes, shapes_edge_attr),
            batch_size,
            model_comm_group,
        )

        return x
