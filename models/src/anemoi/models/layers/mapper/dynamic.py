# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
from typing import Optional

import torch
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData
from torch_geometric.typing import PairTensor

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import change_channels_in_shape
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.block import GraphTransformerMapperBlock
from anemoi.models.layers.mapper.base import BackwardMapperPostProcessMixin
from anemoi.models.layers.mapper.base import BaseMapper
from anemoi.models.layers.mapper.base import ForwardMapperPreProcessMixin
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class DynamicGraphTransformerBaseMapper(BaseMapper):
    """Dynamic Graph Transformer Base Mapper from hidden -> data or data -> hidden."""

    def __init__(
        self,
        in_channels_src: int = 0,
        in_channels_dst: int = 0,
        hidden_dim: int = 128,
        out_channels_dst: Optional[int] = None,
        subgraph_edge_attributes: Optional[list] = [],
        subgraph_edge_index_name: str = "edge_index",
        layer_kernels: DotDict = None,
        num_chunks: int = 1,
        cpu_offload: bool = False,
        num_heads: int = 16,
        mlp_hidden_ratio: int = 4,
        edge_dim: int = 0,
    ) -> None:
        """Initialize DynamicGraphTransformerBaseMapper.

        Parameters
        ----------
        in_channels_src : int
            Input channels of the source node
        in_channels_dst : int
            Input channels of the destination node
        hidden_dim : int
            Hidden dimension
        num_heads: int
            Number of heads to use, default 16
        mlp_hidden_ratio: int
            ratio of mlp hidden dimension to embedding dimension, default 4
        subgraph_edge_attributes: list[str]
            Names of edge attributes to consider
        subgraph_edge_index_name: str
            Name of the edge index attribute in the graph. Defaults to "edge_index".
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        out_channels_dst : Optional[int], optional
            Output channels of the destination node, by default None
        edge_dim : int, optional
            The dimension of the edge attributes
        """
        super().__init__(
            in_channels_src=in_channels_src,
            in_channels_dst=in_channels_dst,
            hidden_dim=hidden_dim,
            out_channels_dst=out_channels_dst,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            layer_kernels=layer_kernels,
        )
        self.edge_attribute_names = subgraph_edge_attributes
        self.edge_index_name = subgraph_edge_index_name

        self.proc = GraphTransformerMapperBlock(
            in_channels=hidden_dim,
            hidden_dim=mlp_hidden_ratio * hidden_dim,
            out_channels=hidden_dim,
            num_heads=num_heads,
            edge_dim=edge_dim,
            num_chunks=num_chunks,
            layer_kernels=self.layer_factory,
        )

        self.offload_layers(cpu_offload)

        self.emb_nodes_dst = (
            nn.Linear(self.in_channels_dst, self.hidden_dim)
            if self.in_channels_dst != self.hidden_dim
            else nn.Identity()
        )

    def forward(
        self,
        x: PairTensor,
        subgraph: HeteroData,
        batch_size: int,
        shard_shapes: tuple[tuple[int], tuple[int]],
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> PairTensor:
        size = (sum(x[0] for x in shard_shapes[0]), sum(x[0] for x in shard_shapes[1]))
        edge_index = subgraph[self.edge_index_name].to(torch.int64)
        edge_attr = torch.cat([subgraph[attr] for attr in self.edge_attribute_names], axis=1)

        shapes_edge_attr = get_shard_shapes(edge_attr, 0, model_comm_group)
        edge_attr = shard_tensor(edge_attr, 0, shapes_edge_attr, model_comm_group)

        x_src, x_dst, shapes_src, shapes_dst = self.pre_process(x, shard_shapes, model_comm_group)

        (x_src, x_dst), edge_attr = self.proc(
            (x_src, x_dst),
            edge_attr,
            edge_index,
            (shapes_src, shapes_dst, shapes_edge_attr),
            batch_size=batch_size,
            model_comm_group=model_comm_group,
            size=size,
        )

        x_dst = self.post_process(x_dst, shapes_dst, model_comm_group)

        return x_dst


class DynamicGraphTransformerForwardMapper(ForwardMapperPreProcessMixin, DynamicGraphTransformerBaseMapper):
    """Dynamic Graph Transformer Mapper from data -> hidden."""

    def __init__(
        self,
        in_channels_src: int = 0,
        in_channels_dst: int = 0,
        hidden_dim: int = 128,
        out_channels_dst: Optional[int] = None,
        subgraph_edge_attributes: Optional[list] = [],
        subgraph_edge_index_name: str = "edge_index",
        layer_kernels: DotDict = None,
        num_chunks: int = 1,
        cpu_offload: bool = False,
        num_heads: int = 16,
        mlp_hidden_ratio: int = 4,
        edge_dim: int = 0,
        **kwargs,
    ) -> None:
        """Initialize DynamicGraphTransformerForwardMapper.

        Parameters
        ----------
        in_channels_src : int
            Input channels of the source node
        in_channels_dst : int
            Input channels of the destination node
        hidden_dim : int
            Hidden dimension
        num_heads: int
            Number of heads to use, default 16
        mlp_hidden_ratio: int
            ratio of mlp hidden dimension to embedding dimension, default 4
        subgraph_edge_attributes: list[str]
            Names of edge attributes to consider
        subgraph_edge_index_name: str
            Name of the edge index attribute in the graph. Defaults to "edge_index".
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        out_channels_dst : Optional[int], optional
            Output channels of the destination node, by default None
        edge_dim: int, optional
            Dimension of the edge attributes
        """
        super().__init__(
            in_channels_src,
            in_channels_dst,
            hidden_dim,
            out_channels_dst=out_channels_dst,
            subgraph_edge_attributes=subgraph_edge_attributes,
            subgraph_edge_index_name=subgraph_edge_index_name,
            layer_kernels=layer_kernels,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            edge_dim=edge_dim,
        )

        self.emb_nodes_src = (
            nn.Linear(self.in_channels_src, self.hidden_dim)
            if self.in_channels_src != self.hidden_dim
            else nn.Identity()
        )

    def forward(
        self,
        x: PairTensor,
        subgraph: HeteroData,
        batch_size: int,
        shard_shapes: tuple[tuple[int], tuple[int]],
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> PairTensor:
        x_dst = super().forward(
            x, subgraph, batch_size=batch_size, shard_shapes=shard_shapes, model_comm_group=model_comm_group
        )
        return x[0], x_dst


class DynamicGraphTransformerBackwardMapper(BackwardMapperPostProcessMixin, DynamicGraphTransformerBaseMapper):
    """Dynamic Graph Transformer Mapper from hidden -> data."""

    def __init__(
        self,
        in_channels_src: int = 0,
        in_channels_dst: int = 0,
        hidden_dim: int = 128,
        out_channels_dst: Optional[int] = None,
        subgraph_edge_attributes: Optional[list] = [],
        subgraph_edge_index_name: str = "edge_index",
        layer_kernels: DotDict = None,
        num_chunks: int = 1,
        cpu_offload: bool = False,
        num_heads: int = 16,
        mlp_hidden_ratio: int = 4,
        edge_dim: int = 0,
        **kwargs,
    ) -> None:
        """Initialize DynamicGraphTransformerBackwardMapper.

        Parameters
        ----------
        in_channels_src : int
            Input channels of the source node
        in_channels_dst : int
            Input channels of the destination node
        hidden_dim : int
            Hidden dimension
        num_heads: int
            Number of heads to use, default 16
        mlp_hidden_ratio: int
            ratio of mlp hidden dimension to embedding dimension, default 4
        subgraph_edge_attributes: list[str]
            Names of edge attributes to consider
        subgraph_edge_index_name: str
            Name of the edge index attribute in the graph. Defaults to "edge_index".
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        out_channels_dst : Optional[int], optional
            Output channels of the destination node, by default None
        edge_dim: int, optional
            Dimension of the edge attributes
        """
        super().__init__(
            in_channels_src,
            in_channels_dst,
            hidden_dim,
            out_channels_dst=out_channels_dst,
            subgraph_edge_attributes=subgraph_edge_attributes,
            subgraph_edge_index_name=subgraph_edge_index_name,
            layer_kernels=layer_kernels,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            edge_dim=edge_dim,
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
