# (C) Copyright 2025 Anemoi contributors.
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
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.chunk import GNNProcessorChunk
from anemoi.models.layers.chunk import GraphTransformerProcessorChunk
from anemoi.models.layers.chunk import PointWiseMLPProcessorChunk
from anemoi.models.layers.chunk import TransformerProcessorChunk
from anemoi.models.layers.graph import TrainableTensor
from anemoi.models.layers.mapper import GraphEdgeMixin
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict


class BaseProcessor(nn.Module, ABC):
    """Base Processor."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_channels: int,
        num_chunks: int,
        cpu_offload: bool = False,
        layer_kernels: DotDict,
        **kwargs,
    ) -> None:
        """Initialize BaseProcessor."""
        super().__init__()

        # Each Processor divides the layers into chunks that get assigned to each ProcessorChunk
        self.num_chunks = num_chunks
        self.num_channels = num_channels
        self.chunk_size = num_layers // num_chunks

        self.layer_factory = load_layer_kernels(layer_kernels)

        self._has_dropout = kwargs.get("dropout_p", 0.0) > 0 if "dropout_p" in kwargs else False

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

        if (model_comm_group := kwargs.get("model_comm_group", None)) is not None:
            assert (
                model_comm_group.size() == 1 or not self._has_dropout
            ), f"Dropout is not supported when model is sharded across {model_comm_group.size()} GPUs"

        x = self.run_layers((x,), *args, **kwargs)
        return x


class PointWiseMLPProcessor(BaseProcessor):
    """Point-wise MLP Processor."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_channels: int,
        num_chunks: int,
        mlp_hidden_ratio: int,
        cpu_offload: bool = False,
        dropout_p: float = 0.0,
        layer_kernels: DotDict,
        **kwargs,
    ):
        super().__init__(
            num_layers=num_layers,
            num_channels=num_channels,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            layer_kernels=layer_kernels,
            dropout_p=dropout_p,
        )

        self.build_layers(
            PointWiseMLPProcessorChunk,
            num_channels=num_channels,
            num_layers=self.chunk_size,
            layer_kernels=self.layer_factory,
            mlp_hidden_ratio=mlp_hidden_ratio,
            dropout_p=dropout_p,
        )

        self.offload_layers(cpu_offload)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_shapes: list[list[int]],
        model_comm_group: Optional[ProcessGroup] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        shape_nodes = change_channels_in_shape(shard_shapes, self.num_channels)
        if model_comm_group:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), f"Only batch size of 1 is supported when model is sharded accross {model_comm_group.size()} GPUs"

        (x,) = self.run_layers((x,), shape_nodes, batch_size, model_comm_group, **kwargs)

        return x


class TransformerProcessor(BaseProcessor):
    """Transformer Processor."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_channels: int,
        num_chunks: int,
        num_heads: int,
        mlp_hidden_ratio: int,
        qk_norm=False,
        dropout_p: float = 0.0,
        attention_implementation: str = "flash_attention",
        softcap: float = 0,
        use_alibi_slopes: bool = False,
        window_size: Optional[int] = None,
        cpu_offload: bool = False,
        layer_kernels: DotDict,
        **kwargs,
    ) -> None:
        """Initialize TransformerProcessor.

        Parameters
        ----------
        num_layers : int
            Number of layers
        num_channels : int
            Number of channels
        num_chunks: int
            Number of chunks in processor
        num_heads: int
            Number of heads in transformer
        mlp_hidden_ratio: int
            Ratio of mlp hidden dimension to embedding dimension
        qk_norm: bool, optional
            Normalize query and key, by default False
        dropout_p: float, optional
            Dropout probability used for multi-head self attention, default 0.1
        attention_implementation: str
            A predefined string which selects which underlying attention
            implementation, by default "flash_attention"
        softcap : float, optional
            Anything > 0 activates softcapping flash attention, by default 0
        use_alibi_slopes : bool
            Use aLiBI option, only used for flash attention, by default False
        window_size: int, optional
            1/2 size of shifted window for attention computation, by default None
        cpu_offload : bool
            Whether to offload processing to CPU, by default False
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
        """
        super().__init__(
            num_layers=num_layers,
            num_channels=num_channels,
            window_size=window_size,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            layer_kernels=layer_kernels,
            dropout_p=dropout_p,
        )

        self.build_layers(
            TransformerProcessorChunk,
            num_channels=num_channels,
            num_layers=self.chunk_size,
            layer_kernels=self.layer_factory,
            mlp_hidden_ratio=mlp_hidden_ratio,
            num_heads=num_heads,
            window_size=window_size,
            qk_norm=qk_norm,
            dropout_p=dropout_p,
            attention_implementation=attention_implementation,
            softcap=softcap,
            use_alibi_slopes=use_alibi_slopes,
        )

        self.offload_layers(cpu_offload)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_shapes: list[list[int]],
        model_comm_group: Optional[ProcessGroup] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        shape_nodes = change_channels_in_shape(shard_shapes, self.num_channels)
        if model_comm_group:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), "Only batch size of 1 is supported when model is sharded accross GPUs"

        (x,) = self.run_layers((x,), shape_nodes, batch_size, model_comm_group, **kwargs)

        return x


class GNNProcessor(GraphEdgeMixin, BaseProcessor):
    """GNN Processor."""

    def __init__(
        self,
        *,
        num_channels: int,
        num_layers: int,
        num_chunks: int,
        mlp_extra_layers: int,
        trainable_size: int,
        src_grid_size: int,
        dst_grid_size: int,
        sub_graph: HeteroData,
        sub_graph_edge_attributes: list[str],
        cpu_offload: bool = False,
        layer_kernels: DotDict,
        **kwargs,
    ) -> None:
        """Initialize GNNProcessor.

        Parameters
        ----------
        num_layers : int
            Number of layers
        num_channels : int
            Number of channels
        num_chunks: int
            Number of chunks in processor
        mlp_extra_layers : int, optional
            Number of extra layers in MLP
        trainable_size : int
            Size of trainable tensor
        src_grid_size : int
            Source grid size
        dst_grid_size : int
            Destination grid size
        sub_graph : HeteroData
            Graph for sub graph in GNN
        sub_graph_edge_attributes : list[str]
            Sub graph edge attributes
        cpu_offload : bool
            Whether to offload processing to CPU, by default False
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml

        """
        super().__init__(
            num_channels=num_channels,
            num_layers=num_layers,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            mlp_extra_layers=mlp_extra_layers,
            layer_kernels=layer_kernels,
        )

        self._register_edges(sub_graph, sub_graph_edge_attributes, src_grid_size, dst_grid_size, trainable_size)

        self.trainable = TrainableTensor(trainable_size=trainable_size, tensor_size=self.edge_attr.shape[0])

        kwargs = {
            "mlp_extra_layers": mlp_extra_layers,
            "layer_kernels": self.layer_factory,
            "edge_dim": None,
        }

        self.build_layers(GNNProcessorChunk, num_channels, self.chunk_size, **kwargs)

        kwargs["edge_dim"] = self.edge_dim  # Edge dim for first layer
        self.proc[0] = GNNProcessorChunk(num_channels, self.chunk_size, **kwargs)

        self.offload_layers(cpu_offload)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_shapes: list[list[int]],
        model_comm_group: Optional[ProcessGroup] = None,
        *args,
        **kwargs,
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

        x, edge_attr = self.run_layers(
            (x, edge_attr), edge_index, (shape_nodes, shape_nodes), model_comm_group, **kwargs
        )

        return x


class GraphTransformerProcessor(GraphEdgeMixin, BaseProcessor):
    """Processor."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_channels: int,
        num_chunks: int,
        num_heads: int,
        mlp_hidden_ratio: int,
        trainable_size: int,
        src_grid_size: int,
        dst_grid_size: int,
        sub_graph: HeteroData,
        sub_graph_edge_attributes: list[str],
        qk_norm: bool = False,
        cpu_offload: bool = False,
        layer_kernels: DotDict,
        **kwargs,
    ) -> None:
        """Initialize GraphTransformerProcessor.

        Parameters
        ----------
        num_layers : int
            Number of layers
        num_channels : int
            Number of channels
        num_chunks: int
            Number of chunks in processor
        num_heads: int
            Number of heads in transformer
        mlp_hidden_ratio: int
            Ratio of mlp hidden dimension to embedding dimension
        trainable_size : int
            Size of trainable tensor
        src_grid_size : int
            Source grid size
        dst_grid_size : int
            Destination grid size
        sub_graph : HeteroData
            Graph for sub graph in GNN
        sub_graph_edge_attributes : list[str]
            Sub graph edge attributes
        qk_norm: bool, optional
            Normalize query and key, by default False
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
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

        self._register_edges(sub_graph, sub_graph_edge_attributes, src_grid_size, dst_grid_size, trainable_size)

        self.trainable = TrainableTensor(trainable_size=trainable_size, tensor_size=self.edge_attr.shape[0])

        self.build_layers(
            GraphTransformerProcessorChunk,
            num_channels=num_channels,
            num_layers=self.chunk_size,
            layer_kernels=self.layer_factory,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            qk_norm=qk_norm,
            edge_dim=self.edge_dim,
        )

        self.offload_layers(cpu_offload)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_shapes: list[list[int]],
        model_comm_group: Optional[ProcessGroup] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        size = sum(x[0] for x in shard_shapes)

        shape_nodes = change_channels_in_shape(shard_shapes, self.num_channels)
        edge_attr = self.trainable(self.edge_attr, batch_size)

        edge_index = self._expand_edges(self.edge_index_base, self.edge_inc, batch_size)

        shapes_edge_attr = get_shard_shapes(edge_attr, 0, model_comm_group)
        edge_attr = shard_tensor(edge_attr, 0, shapes_edge_attr, model_comm_group)

        x, edge_attr = self.run_layers(
            data=(x, edge_attr),
            edge_index=edge_index,
            shapes=(shape_nodes, shape_nodes, shapes_edge_attr),
            batch_size=batch_size,
            size=size,
            model_comm_group=model_comm_group,
            **kwargs,
        )

        return x
