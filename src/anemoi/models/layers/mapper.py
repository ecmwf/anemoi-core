# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
from abc import ABC
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData
from torch_geometric.typing import Adj
from torch_geometric.typing import PairTensor

from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.khop_edges import sort_edges_1hop_sharding
from anemoi.models.distributed.shapes import change_channels_in_shape
from anemoi.models.distributed.shapes import get_shape_shards
from anemoi.models.layers.block import GraphConvMapperBlock
from anemoi.models.layers.block import GraphTransformerMapperBlock
from anemoi.models.layers.graph import TrainableTensor
from anemoi.models.layers.mlp import MLP

LOGGER = logging.getLogger(__name__)


class BaseMapper(nn.Module, ABC):
    """Base Mapper from souce dimension to destination dimension."""

    def __init__(
        self,
        in_channels_src: int = 0,
        in_channels_dst: int = 0,
        hidden_dim: int = 128,
        out_channels_dst: Optional[int] = None,
        cpu_offload: bool = False,
        activation: str = "SiLU",
        **kwargs,
    ) -> None:
        """Initialize BaseMapper."""
        super().__init__()

        self.in_channels_src = in_channels_src
        self.in_channels_dst = in_channels_dst
        self.hidden_dim = hidden_dim
        self.out_channels_dst = out_channels_dst
        self.activation = activation

        self.proc = NotImplemented

        self.offload_layers(cpu_offload)

    def offload_layers(self, cpu_offload):
        if cpu_offload:
            self.proc = nn.ModuleList([offload_wrapper(x) for x in self.proc])

    def pre_process(self, x, shard_shapes, model_comm_group=None) -> tuple[Tensor, Tensor, tuple[int], tuple[int]]:
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

    def post_process(self, x_dst, shapes_dst, model_comm_group=None):
        """Post-processing for the mapper."""
        return x_dst


class BackwardMapperPostProcessMixin:
    """Post-processing for Backward Mapper from hidden -> data."""

    def post_process(self, x_dst, shapes_dst, model_comm_group=None):
        x_dst = self.node_data_extractor(x_dst)
        x_dst = gather_tensor(x_dst, 0, change_channels_in_shape(shapes_dst, self.out_channels_dst), model_comm_group)
        return x_dst


class ForwardMapperPreProcessMixin:
    """Pre-processing for Forward Mapper from data -> hidden."""

    def pre_process(self, x, shard_shapes, model_comm_group=None):
        x_src, x_dst, shapes_src, shapes_dst = super().pre_process(x, shard_shapes, model_comm_group)
        x_src = shard_tensor(x_src, 0, shapes_src, model_comm_group)
        x_dst = shard_tensor(x_dst, 0, shapes_dst, model_comm_group)
        x_src = self.emb_nodes_src(x_src)
        x_dst = self.emb_nodes_dst(x_dst)
        shapes_src = change_channels_in_shape(shapes_src, self.hidden_dim)
        shapes_dst = change_channels_in_shape(shapes_dst, self.hidden_dim)
        return x_src, x_dst, shapes_src, shapes_dst


class GraphEdgeMixin:
    def _register_edges(
        self, sub_graph: HeteroData, edge_attributes: list[str], src_size: int, dst_size: int, trainable_size: int
    ) -> None:
        """Register edge dim, attr, index_base, and increment.

        Parameters
        ----------
        sub_graph : HeteroData
            Sub graph of the full structure
        edge_attributes : list[str]
            Edge attributes to use.
        src_size : int
            Source size
        dst_size : int
            Target size
        trainable_size : int
            Trainable tensor size
        """
        assert sub_graph, f"{self.__class__.__name__} needs a valid sub_graph to register edges."
        assert edge_attributes is not None, "Edge attributes must be provided"

        edge_attr_tensor = torch.cat([sub_graph[attr] for attr in edge_attributes], axis=1)

        self.edge_dim = edge_attr_tensor.shape[1] + trainable_size
        self.register_buffer("edge_attr", edge_attr_tensor, persistent=False)
        self.register_buffer("edge_index_base", sub_graph.edge_index, persistent=False)
        self.register_buffer(
            "edge_inc", torch.from_numpy(np.asarray([[src_size], [dst_size]], dtype=np.int64)), persistent=True
        )

    def _expand_edges(self, edge_index: Adj, edge_inc: Tensor, batch_size: int) -> Adj:
        """Expand edge index while incrementing to the edge index.

        Parameters
        ----------
        edge_index : Adj
            Edge index to start
        edge_inc : Tensor
            Edge increment to use
        batch_size : int
            Number of times to expand the edge index

        Returns
        -------
        Tensor
            Edge Index
        """
        edge_index = torch.cat(
            [edge_index + i * edge_inc for i in range(batch_size)],
            dim=1,
        )
        return edge_index


class GraphTransformerBaseMapper(GraphEdgeMixin, BaseMapper):
    """Graph Transformer Base Mapper from hidden -> data or data -> hidden."""

    def __init__(
        self,
        in_channels_src: int = 0,
        in_channels_dst: int = 0,
        hidden_dim: int = 128,
        trainable_size: int = 8,
        out_channels_dst: Optional[int] = None,
        num_chunks: int = 1,
        cpu_offload: bool = False,
        activation: str = "GELU",
        num_heads: int = 16,
        mlp_hidden_ratio: int = 4,
        sub_graph: Optional[HeteroData] = None,
        sub_graph_edge_attributes: Optional[list[str]] = None,
        src_grid_size: int = 0,
        dst_grid_size: int = 0,
    ) -> None:
        """Initialize GraphTransformerBaseMapper.

        Parameters
        ----------
        in_channels_src : int
            Input channels of the source node
        in_channels_dst : int
            Input channels of the destination node
        hidden_dim : int
            Hidden dimension
        trainable_size : int
            Trainable tensor of edge
        num_heads: int
            Number of heads to use, default 16
        mlp_hidden_ratio: int
            ratio of mlp hidden dimension to embedding dimension, default 4
        activation : str, optional
            Activation function, by default "GELU"
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        out_channels_dst : Optional[int], optional
            Output channels of the destination node, by default None
        """
        super().__init__(
            in_channels_src,
            in_channels_dst,
            hidden_dim,
            out_channels_dst=out_channels_dst,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            activation=activation,
        )

        self._register_edges(sub_graph, sub_graph_edge_attributes, src_grid_size, dst_grid_size, trainable_size)

        self.trainable = TrainableTensor(trainable_size=trainable_size, tensor_size=self.edge_attr.shape[0])

        self.proc = GraphTransformerMapperBlock(
            hidden_dim,
            mlp_hidden_ratio * hidden_dim,
            hidden_dim,
            num_heads=num_heads,
            edge_dim=self.edge_dim,
            activation=activation,
            num_chunks=num_chunks,
        )

        self.offload_layers(cpu_offload)

        self.emb_nodes_dst = nn.Linear(self.in_channels_dst, self.hidden_dim)

    def forward(
        self,
        x: PairTensor,
        batch_size: int,
        shard_shapes: tuple[tuple[int], tuple[int]],
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> PairTensor:
        size = (sum(x[0] for x in shard_shapes[0]), sum(x[0] for x in shard_shapes[1]))
        edge_attr = self.trainable(self.edge_attr, batch_size)
        edge_index = self._expand_edges(self.edge_index_base, self.edge_inc, batch_size)
        shapes_edge_attr = get_shape_shards(edge_attr, 0, model_comm_group)
        edge_attr = shard_tensor(edge_attr, 0, shapes_edge_attr, model_comm_group)

        x_src, x_dst, shapes_src, shapes_dst = self.pre_process(x, shard_shapes, model_comm_group)

        (x_src, x_dst), edge_attr = self.proc(
            (x_src, x_dst),
            edge_attr,
            edge_index,
            (shapes_src, shapes_dst, shapes_edge_attr),
            batch_size,
            model_comm_group,
            size=size,
        )

        x_dst = self.post_process(x_dst, shapes_dst, model_comm_group)

        return x_dst


class GraphTransformerForwardMapper(ForwardMapperPreProcessMixin, GraphTransformerBaseMapper):
    """Graph Transformer Mapper from data -> hidden."""

    def __init__(
        self,
        in_channels_src: int = 0,
        in_channels_dst: int = 0,
        hidden_dim: int = 128,
        trainable_size: int = 8,
        out_channels_dst: Optional[int] = None,
        num_chunks: int = 1,
        cpu_offload: bool = False,
        activation: str = "GELU",
        num_heads: int = 16,
        mlp_hidden_ratio: int = 4,
        sub_graph: Optional[HeteroData] = None,
        sub_graph_edge_attributes: Optional[list[str]] = None,
        src_grid_size: int = 0,
        dst_grid_size: int = 0,
    ) -> None:
        """Initialize GraphTransformerForwardMapper.

        Parameters
        ----------
        in_channels_src : int
            Input channels of the source node
        in_channels_dst : int
            Input channels of the destination node
        hidden_dim : int
            Hidden dimension
        trainable_size : int
            Trainable tensor of edge
        num_heads: int
            Number of heads to use, default 16
        mlp_hidden_ratio: int
            ratio of mlp hidden dimension to embedding dimension, default 4
        activation : str, optional
            Activation function, by default "GELU"
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        out_channels_dst : Optional[int], optional
            Output channels of the destination node, by default None
        """
        super().__init__(
            in_channels_src,
            in_channels_dst,
            hidden_dim,
            trainable_size,
            out_channels_dst=out_channels_dst,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            activation=activation,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            sub_graph=sub_graph,
            sub_graph_edge_attributes=sub_graph_edge_attributes,
            src_grid_size=src_grid_size,
            dst_grid_size=dst_grid_size,
        )

        self.emb_nodes_src = nn.Linear(self.in_channels_src, self.hidden_dim)

    def forward(
        self,
        x: PairTensor,
        batch_size: int,
        shard_shapes: tuple[tuple[int], tuple[int]],
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> PairTensor:
        x_dst = super().forward(x, batch_size, shard_shapes, model_comm_group)
        return x[0], x_dst


class GraphTransformerBackwardMapper(BackwardMapperPostProcessMixin, GraphTransformerBaseMapper):
    """Graph Transformer Mapper from hidden -> data."""

    def __init__(
        self,
        in_channels_src: int = 0,
        in_channels_dst: int = 0,
        hidden_dim: int = 128,
        trainable_size: int = 8,
        out_channels_dst: Optional[int] = None,
        num_chunks: int = 1,
        cpu_offload: bool = False,
        activation: str = "GELU",
        num_heads: int = 16,
        mlp_hidden_ratio: int = 4,
        sub_graph: Optional[HeteroData] = None,
        sub_graph_edge_attributes: Optional[list[str]] = None,
        src_grid_size: int = 0,
        dst_grid_size: int = 0,
    ) -> None:
        """Initialize GraphTransformerBackwardMapper.

        Parameters
        ----------
        in_channels_src : int
            Input channels of the source node
        in_channels_dst : int
            Input channels of the destination node
        hidden_dim : int
            Hidden dimension
        trainable_size : int
            Trainable tensor of edge
        num_heads: int
            Number of heads to use, default 16
        mlp_hidden_ratio: int
            ratio of mlp hidden dimension to embedding dimension, default 4
        activation : str, optional
            Activation function, by default "GELU"
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        out_channels_dst : Optional[int], optional
            Output channels of the destination node, by default None
        """
        super().__init__(
            in_channels_src,
            in_channels_dst,
            hidden_dim,
            trainable_size,
            out_channels_dst=out_channels_dst,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            activation=activation,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            sub_graph=sub_graph,
            sub_graph_edge_attributes=sub_graph_edge_attributes,
            src_grid_size=src_grid_size,
            dst_grid_size=dst_grid_size,
        )

        self.node_data_extractor = nn.Sequential(
            nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.out_channels_dst)
        )

    def pre_process(self, x, shard_shapes, model_comm_group=None):
        x_src, x_dst, shapes_src, shapes_dst = super().pre_process(x, shard_shapes, model_comm_group)
        shapes_src = change_channels_in_shape(shapes_src, self.hidden_dim)
        x_dst = shard_tensor(x_dst, 0, shapes_dst, model_comm_group)
        x_dst = self.emb_nodes_dst(x_dst)
        shapes_dst = change_channels_in_shape(shapes_dst, self.hidden_dim)
        return x_src, x_dst, shapes_src, shapes_dst


class GNNBaseMapper(GraphEdgeMixin, BaseMapper):
    """Base for Graph Neural Network Mapper from hidden -> data or data -> hidden."""

    def __init__(
        self,
        in_channels_src: int = 0,
        in_channels_dst: int = 0,
        hidden_dim: int = 128,
        trainable_size: int = 8,
        out_channels_dst: Optional[int] = None,
        num_chunks: int = 1,
        cpu_offload: bool = False,
        activation: str = "SiLU",
        mlp_extra_layers: int = 0,
        sub_graph: Optional[HeteroData] = None,
        sub_graph_edge_attributes: Optional[list[str]] = None,
        src_grid_size: int = 0,
        dst_grid_size: int = 0,
    ) -> None:
        """Initialize GNNBaseMapper.

        Parameters
        ----------
        in_channels_src : int
            Input channels of the source node
        in_channels_dst : int
            Input channels of the destination node
        hidden_dim : int
            Hidden dimension
        trainable_size : int
            Trainable tensor of edge
        mlp_extra_layers : int, optional
            Number of extra layers in MLP, by default 0
        activation : str, optional
            Activation function, by default "SiLU"
        num_chunks : int
            Do message passing in X chunks
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        out_channels_dst : Optional[int], optional
            Output channels of the destination node, by default None
        """
        super().__init__(
            in_channels_src,
            in_channels_dst,
            hidden_dim,
            out_channels_dst=out_channels_dst,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            activation=activation,
        )

        self._register_edges(sub_graph, sub_graph_edge_attributes, src_grid_size, dst_grid_size, trainable_size)

        self.emb_edges = MLP(
            in_features=self.edge_dim,
            hidden_dim=hidden_dim,
            out_features=hidden_dim,
            n_extra_layers=mlp_extra_layers,
            activation=activation,
        )

        self.trainable = TrainableTensor(trainable_size=trainable_size, tensor_size=self.edge_attr.shape[0])

    def prepare_edges(self, size, batch_size, model_comm_group=None):
        edge_attr = self.trainable(self.edge_attr, batch_size)
        edge_index = self._expand_edges(self.edge_index_base, self.edge_inc, batch_size)
        edge_attr, edge_index, shapes_edge_attr, shapes_edge_idx = sort_edges_1hop_sharding(
            size, edge_attr, edge_index, model_comm_group
        )

        edge_index = shard_tensor(edge_index, 1, shapes_edge_idx, model_comm_group)
        edge_attr = shard_tensor(edge_attr, 0, shapes_edge_attr, model_comm_group)
        edge_attr = self.emb_edges(edge_attr)
        return edge_attr, edge_index

    def forward(
        self,
        x: PairTensor,
        batch_size: int,
        shard_shapes: tuple[tuple[int], tuple[int]],
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> PairTensor:

        size = (sum(x[0] for x in shard_shapes[0]), sum(x[0] for x in shard_shapes[1]))

        edge_attr, edge_index = self.prepare_edges(size, batch_size, model_comm_group)

        x_src, x_dst, shapes_src, shapes_dst = self.pre_process(x, shard_shapes, model_comm_group)

        (x_src, x_dst), edge_attr = self.proc(
            (x_src, x_dst),
            edge_attr,
            edge_index,
            (shapes_src, shapes_dst),
            model_comm_group,
            size=size,
        )

        x_dst = self.post_process(x_dst, shapes_dst, model_comm_group)

        return x_src, x_dst


class GNNForwardMapper(ForwardMapperPreProcessMixin, GNNBaseMapper):
    """Graph Neural Network Mapper data -> hidden."""

    def __init__(
        self,
        in_channels_src: int = 0,
        in_channels_dst: int = 0,
        hidden_dim: int = 128,
        trainable_size: int = 8,
        out_channels_dst: Optional[int] = None,
        num_chunks: int = 1,
        cpu_offload: bool = False,
        activation: str = "SiLU",
        mlp_extra_layers: int = 0,
        sub_graph: Optional[HeteroData] = None,
        sub_graph_edge_attributes: Optional[list[str]] = None,
        src_grid_size: int = 0,
        dst_grid_size: int = 0,
    ) -> None:
        """Initialize GNNForwardMapper.

        Parameters
        ----------
        in_channels_src : int
            Input channels of the source node
        in_channels_dst : int
            Input channels of the destination node
        hidden_dim : int
            Hidden dimension
        edge_dim : int
            Trainable tensor of edge
        mlp_extra_layers : int, optional
            Number of extra layers in MLP, by default 0
        activation : str, optional
            Activation function, by default "SiLU"
        num_chunks : int
            Do message passing in X chunks
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        out_channels_dst : Optional[int], optional
            Output channels of the destination node, by default None
        """
        super().__init__(
            in_channels_src,
            in_channels_dst,
            hidden_dim,
            trainable_size,
            out_channels_dst,
            num_chunks,
            cpu_offload,
            activation,
            mlp_extra_layers,
            sub_graph=sub_graph,
            sub_graph_edge_attributes=sub_graph_edge_attributes,
            src_grid_size=src_grid_size,
            dst_grid_size=dst_grid_size,
        )

        self.proc = GraphConvMapperBlock(
            hidden_dim,
            hidden_dim,
            mlp_extra_layers=mlp_extra_layers,
            activation=activation,
            update_src_nodes=True,
            num_chunks=num_chunks,
        )

        self.offload_layers(cpu_offload)

        self.emb_nodes_src = MLP(
            in_features=in_channels_src,
            hidden_dim=hidden_dim,
            out_features=hidden_dim,
            n_extra_layers=mlp_extra_layers,
            activation=activation,
        )

        self.emb_nodes_dst = MLP(
            in_features=in_channels_dst,
            hidden_dim=hidden_dim,
            out_features=hidden_dim,
            n_extra_layers=mlp_extra_layers,
            activation=activation,
        )


class GNNBackwardMapper(BackwardMapperPostProcessMixin, GNNBaseMapper):
    """Graph Neural Network Mapper from hidden -> data."""

    def __init__(
        self,
        in_channels_src: int = 0,
        in_channels_dst: int = 0,
        hidden_dim: int = 128,
        trainable_size: int = 8,
        out_channels_dst: Optional[int] = None,
        num_chunks: int = 1,
        cpu_offload: bool = False,
        activation: str = "SiLU",
        mlp_extra_layers: int = 0,
        sub_graph: Optional[HeteroData] = None,
        sub_graph_edge_attributes: Optional[list[str]] = None,
        src_grid_size: int = 0,
        dst_grid_size: int = 0,
    ) -> None:
        """Initialize GNNBackwardMapper.

        Parameters
        ----------
        in_channels_src : int
            Input channels of the source node
        in_channels_dst : int
            Input channels of the destination node
        hidden_dim : int
            Hidden dimension
        edge_dim : int
            Trainable tensor of edge
        mlp_extra_layers : int, optional
            Number of extra layers in MLP, by default 0
        activation : str, optional
            Activation function, by default "SiLU"
        num_chunks : int
            Do message passing in X chunks
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        out_channels_dst : Optional[int], optional
            Output channels of the destination node, by default None
        """
        super().__init__(
            in_channels_src,
            in_channels_dst,
            hidden_dim,
            trainable_size,
            out_channels_dst=out_channels_dst,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            activation=activation,
            mlp_extra_layers=mlp_extra_layers,
            sub_graph=sub_graph,
            sub_graph_edge_attributes=sub_graph_edge_attributes,
            src_grid_size=src_grid_size,
            dst_grid_size=dst_grid_size,
        )

        self.proc = GraphConvMapperBlock(
            hidden_dim,
            hidden_dim,
            mlp_extra_layers=mlp_extra_layers,
            activation=activation,
            update_src_nodes=False,
            num_chunks=num_chunks,
        )

        self.offload_layers(cpu_offload)

        self.node_data_extractor = MLP(
            in_features=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            out_features=self.out_channels_dst,
            n_extra_layers=mlp_extra_layers,
            activation=self.activation,
            layer_norm=False,
            final_activation=False,
        )

    def pre_process(self, x, shard_shapes, model_comm_group=None):
        x_src, x_dst, shapes_src, shapes_dst = super().pre_process(x, shard_shapes, model_comm_group)
        shapes_src = change_channels_in_shape(shapes_src, self.hidden_dim)
        shapes_dst = change_channels_in_shape(shapes_dst, self.hidden_dim)
        return x_src, x_dst, shapes_src, shapes_dst

    def forward(
        self,
        x: PairTensor,
        batch_size: int,
        shard_shapes: tuple[tuple[int], tuple[int]],
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> Tensor:

        _, x_dst = super().forward(x, batch_size, shard_shapes, model_comm_group)
        return x_dst
