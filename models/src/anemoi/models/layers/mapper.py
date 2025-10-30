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

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData
from torch_geometric.typing import Adj
from torch_geometric.typing import PairTensor

from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.graph import sync_tensor
from anemoi.models.distributed.khop_edges import bipartite_subgraph
from anemoi.models.distributed.khop_edges import drop_unconnected_src_nodes
from anemoi.models.distributed.khop_edges import sort_edges_1hop_sharding
from anemoi.models.distributed.shapes import change_channels_in_shape
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.block import GraphConvMapperBlock
from anemoi.models.layers.block import GraphTransformerMapperBlock
from anemoi.models.layers.block import TransformerMapperBlock
from anemoi.models.layers.graph import TrainableTensor
from anemoi.models.layers.mlp import MLP
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
        shard_shapes : Tuple[List[int], List[int]]
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
        *,
        in_channels_src: int,
        in_channels_dst: int,
        hidden_dim: int,
        out_channels_dst: Optional[int] = None,
        trainable_size: int,
        num_chunks: int,
        num_heads: int,
        mlp_hidden_ratio: int,
        sub_graph: HeteroData,
        sub_graph_edge_attributes: list[str],
        src_grid_size: int,
        dst_grid_size: int,
        qk_norm: bool = False,
        cpu_offload: bool = False,
        layer_kernels: DotDict = None,
        shard_strategy: str = "edges",
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
        out_channels_dst : int, optional
            Output channels of the destination node, by default None
        trainable_size : int
            Trainable tensor of edge
        num_chunks : int
            Number of chunks to split into
        num_heads: int
            Number of heads in transformer
        mlp_hidden_ratio: int
            ratio of mlp hidden dimension to embedding dimension
        sub_graph : HeteroData
            Sub graph of the full structure
        sub_graph_edge_attributes : list[str]
            Edge attributes to use
        src_grid_size : int
            Source grid size
        dst_grid_size : int
            Destination grid size
        qk_norm : bool, optional
            Whether to use query and key normalization, default False
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        layer_kernels : DotDict, optional
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
        shard_strategy : str, optional
            Strategy to shard tensors, by default "edges"
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

        self.num_chunks = num_chunks

        Linear = self.layer_factory.Linear

        self._register_edges(sub_graph, sub_graph_edge_attributes, src_grid_size, dst_grid_size, trainable_size)

        self.trainable = TrainableTensor(trainable_size=trainable_size, tensor_size=self.edge_attr.shape[0])

        self.proc = GraphTransformerMapperBlock(
            in_channels=hidden_dim,
            hidden_dim=mlp_hidden_ratio * hidden_dim,
            out_channels=hidden_dim,
            num_heads=num_heads,
            edge_dim=self.edge_dim,
            qk_norm=qk_norm,
            layer_kernels=self.layer_factory,
            shard_strategy=shard_strategy,
        )

        self.offload_layers(cpu_offload)

        self.emb_nodes_dst = Linear(self.in_channels_dst, self.hidden_dim)

        self.shard_strategy = shard_strategy

        assert shard_strategy in ["heads", "edges"], (
            f"Invalid shard strategy '{shard_strategy}' for {self.__class__.__name__}. "
            f"Supported strategies are 'heads' and 'edges'."
        )

    def prepare_edges(
        self,
        size: tuple[int, int],
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[Tensor, Adj]:
        edge_attr = self.trainable(self.edge_attr, batch_size)
        edge_index = self._expand_edges(self.edge_index_base, self.edge_inc, batch_size)
        edge_attr, edge_index, shapes_edge_attr, shapes_edge_idx = sort_edges_1hop_sharding(
            size, edge_attr, edge_index, model_comm_group, relabel_dst_nodes=True
        )

        edge_index = shard_tensor(edge_index, 1, shapes_edge_idx, model_comm_group)
        edge_attr = shard_tensor(edge_attr, 0, shapes_edge_attr, model_comm_group)

        return edge_attr, edge_index

    def pre_process_edge_sharding_wrapper(
        self,
        x: PairTensor,
        shard_shapes: tuple[list[list[int]], list[list[int]]],
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
        x_src_is_sharded: bool = False,
        x_dst_is_sharded: bool = False,
        cond: Optional[tuple[Tensor, Tensor]] = None,
    ):
        x_src, x_dst = x
        shapes_src, shapes_dst = shard_shapes

        shapes_x_src = change_channels_in_shape(shapes_src, x_src.shape[-1])
        # gather/scatter if x_src is sharded, always reduce gradients in bwds
        x_src = sync_tensor(x_src, 0, shapes_x_src, model_comm_group, gather_in_fwd=x_src_is_sharded)

        size_full_graph = (sum(shape[0] for shape in shard_shapes[0]), sum(shape[0] for shape in shard_shapes[1]))
        edge_attr, edge_index = self.prepare_edges(size_full_graph, batch_size, model_comm_group)

        # at this point, x_src is synced i.e. full, x_dst is sharded, edges are sharded (incoming edges to x_dst)
        size_src_full_dst_shard = (x_src.shape[0], x_dst.shape[0])
        x_src, edge_index, nodes_src = drop_unconnected_src_nodes(x_src, edge_index, size_src_full_dst_shard)

        if cond is not None:  # sync cond_src to match x_src:
            cond_src, cond_dst = cond
            shapes_cond_src = change_channels_in_shape(shapes_src, cond_src.shape[-1])
            cond_src_full = sync_tensor(cond_src, 0, shapes_cond_src, model_comm_group, gather_in_fwd=True)
            cond = (cond_src_full[nodes_src], cond_dst)

        if not x_dst_is_sharded:
            x_dst = shard_tensor(x_dst, 0, shapes_dst, model_comm_group)

        return x_src, x_dst, edge_attr, edge_index, shapes_src, shapes_dst, cond

    def run_processor_chunk_edge_sharding(
        self,
        x: tuple[Tensor, Tensor],
        dst_chunk: Tensor,
        edge_attr: Tensor,
        edge_index: Adj,
        shapes: tuple[tuple[int], tuple[int]],
        batch_size: int,
        size: tuple[int],
        model_comm_group: Optional[ProcessGroup] = None,
        cond: Optional[tuple[Tensor, Tensor]] = None,
        **kwargs,
    ) -> Tensor:
        x_src, x_dst = x

        # get subgraph of x_dst_chunk and incoming edges, drop unconnected src nodes
        nodes_src_full = torch.arange(size[0], device=edge_index.device)
        edge_index, edge_attr = bipartite_subgraph(
            (nodes_src_full, dst_chunk),
            edge_index,
            edge_attr,
            size=size,
            relabel_nodes=True,
        )

        # drop unconnected src nodes and relabel edges
        x_src_chunk, edge_index_chunk, connected_src_nodes = drop_unconnected_src_nodes(x_src, edge_index, size)
        x_dst_chunk = x_dst[dst_chunk]
        chunk_size = (x_src_chunk.shape[0], x_dst_chunk.shape[0])

        if cond is not None:  # update cond with correct conditioning
            cond_src, cond_dst = cond
            cond = (cond_src[connected_src_nodes], cond_dst[dst_chunk])

        # pre-process chunk, embedding x_src/x_dst if not already done
        x_src_chunk, x_dst_chunk, _, _ = self.pre_process(
            (x_src_chunk, x_dst_chunk), shapes, model_comm_group, x_src_is_sharded=True, x_dst_is_sharded=True
        )

        (_, x_dst_out), _ = self.proc(
            (x_src_chunk, x_dst_chunk),
            edge_attr,
            edge_index_chunk,
            shapes,
            batch_size,
            chunk_size,
            model_comm_group,
            cond=cond,
            **kwargs,
        )

        return self.post_process(x_dst_out, shapes[1], model_comm_group, keep_x_dst_sharded=True)

    def mapper_forward_with_edge_sharding(
        self,
        x: PairTensor,
        batch_size: int,
        shard_shapes: tuple[list[list[int]], list[list[int]]],
        model_comm_group: Optional[ProcessGroup] = None,
        x_src_is_sharded: bool = False,
        x_dst_is_sharded: bool = False,
        keep_x_dst_sharded: bool = False,
        cond: Optional[tuple[Tensor, Tensor]] = None,
        **kwargs,
    ) -> PairTensor:
        x_src, x_dst, edge_attr, edge_index, shapes_src, shapes_dst, cond = checkpoint(
            self.pre_process_edge_sharding_wrapper,
            x,
            shard_shapes,
            batch_size,
            model_comm_group,
            x_src_is_sharded,
            x_dst_is_sharded,
            cond,
            use_reentrant=False,
        )

        size = (x_src.shape[0], x_dst.shape[0])  # node sizes of local graph shard
        num_chunks = max(self.num_chunks, NUM_CHUNKS_INFERENCE_MAPPER)

        dst_chunks = torch.arange(size[1], device=x_dst.device).tensor_split(num_chunks)
        out_channels = self.out_channels_dst if self.out_channels_dst is not None else self.hidden_dim
        out_type = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else x_dst.dtype
        out_dst = torch.empty((*x_dst.shape[:-1], out_channels), device=x_dst.device, dtype=out_type)

        for dst_chunk in dst_chunks:
            out_dst[dst_chunk] = checkpoint(
                self.run_processor_chunk_edge_sharding,
                (x_src, x_dst),
                dst_chunk,
                edge_attr,
                edge_index,
                (shapes_src, shapes_dst),
                batch_size,
                size,
                model_comm_group,
                cond,
                **kwargs,
                use_reentrant=False,
            ).to(dtype=out_type)

        if not keep_x_dst_sharded:  # gather after processing chunks
            out_dst = gather_tensor(out_dst, 0, change_channels_in_shape(shapes_dst, out_channels), model_comm_group)

        return out_dst

    def mapper_forward_with_heads_sharding(
        self,
        x: PairTensor,
        batch_size: int,
        shard_shapes: tuple[list[list[int]], list[list[int]]],
        model_comm_group: Optional[ProcessGroup] = None,
        x_src_is_sharded: bool = False,
        x_dst_is_sharded: bool = False,
        keep_x_dst_sharded: bool = False,
        **kwargs,
    ) -> PairTensor:
        size = (sum(x[0] for x in shard_shapes[0]), sum(x[0] for x in shard_shapes[1]))
        edge_attr = self.trainable(self.edge_attr, batch_size)
        edge_index = self._expand_edges(self.edge_index_base, self.edge_inc, batch_size)
        shapes_edge_attr = get_shard_shapes(edge_attr, 0, model_comm_group)
        edge_attr = shard_tensor(edge_attr, 0, shapes_edge_attr, model_comm_group)

        x_src, x_dst, shapes_src, shapes_dst = self.pre_process(
            x, shard_shapes, model_comm_group, x_src_is_sharded, x_dst_is_sharded
        )

        (x_src, x_dst), edge_attr = self.proc(
            x=(x_src, x_dst),
            edge_attr=edge_attr,
            edge_index=edge_index,
            shapes=(shapes_src, shapes_dst, shapes_edge_attr),
            batch_size=batch_size,
            size=size,
            model_comm_group=model_comm_group,
            **kwargs,
        )

        x_dst = self.post_process(x_dst, shapes_dst, model_comm_group, keep_x_dst_sharded=keep_x_dst_sharded)

        return x_dst

    def forward(
        self,
        x: PairTensor,
        batch_size: int,
        shard_shapes: tuple[list[list[int]], list[list[int]]],
        model_comm_group: Optional[ProcessGroup] = None,
        x_src_is_sharded: bool = False,
        x_dst_is_sharded: bool = False,
        keep_x_dst_sharded: bool = False,
        **kwargs,
    ) -> PairTensor:

        kwargs_forward = {
            "x": x,
            "batch_size": batch_size,
            "shard_shapes": shard_shapes,
            "model_comm_group": model_comm_group,
            "x_src_is_sharded": x_src_is_sharded,
            "x_dst_is_sharded": x_dst_is_sharded,
            "keep_x_dst_sharded": keep_x_dst_sharded,
            **kwargs,
        }

        if self.shard_strategy == "edges":
            return self.mapper_forward_with_edge_sharding(**kwargs_forward)
        else:  # self.shard_strategy == "heads"
            return checkpoint(self.mapper_forward_with_heads_sharding, **kwargs_forward, use_reentrant=False)


class GraphTransformerForwardMapper(ForwardMapperPreProcessMixin, GraphTransformerBaseMapper):
    """Graph Transformer Mapper from data -> hidden."""

    def __init__(
        self,
        *,
        in_channels_src: int,
        in_channels_dst: int,
        hidden_dim: int,
        trainable_size: int,
        num_chunks: int,
        num_heads: int,
        mlp_hidden_ratio: int,
        sub_graph: HeteroData,
        sub_graph_edge_attributes: list[str],
        src_grid_size: int,
        dst_grid_size: int,
        qk_norm: bool = False,
        cpu_offload: bool = False,
        layer_kernels: DotDict = None,
        shard_strategy: str = "edges",
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
        num_chunks : int
            Number of chunks to split into
        num_heads: int
            Number of heads in transformer
        mlp_hidden_ratio: int
            ratio of mlp hidden dimension to embedding dimension, default 4
        sub_graph : HeteroData
            Sub graph of the full structure
        sub_graph_edge_attributes : list[str]
            Edge attributes to use
        src_grid_size : int
            Source grid size
        dst_grid_size : int
            Destination grid size
        qk_norm : bool, optional
            Whether to use query and key normalization, default False
        cpu_offload : bool
            Whether to offload processing to CPU, by default False
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
        shard_strategy : str, optional
            Strategy to shard tensors, by default "edges"
        """
        super().__init__(
            in_channels_src=in_channels_src,
            in_channels_dst=in_channels_dst,
            hidden_dim=hidden_dim,
            out_channels_dst=None,
            trainable_size=trainable_size,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            qk_norm=qk_norm,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            sub_graph=sub_graph,
            sub_graph_edge_attributes=sub_graph_edge_attributes,
            src_grid_size=src_grid_size,
            dst_grid_size=dst_grid_size,
            layer_kernels=layer_kernels,
            shard_strategy=shard_strategy,
        )

        self.emb_nodes_src = self.layer_factory.Linear(self.in_channels_src, self.hidden_dim)

    def forward(
        self,
        x: PairTensor,
        batch_size: int,
        shard_shapes: tuple[list[list[int]], list[list[int]]],
        model_comm_group: Optional[ProcessGroup] = None,
        x_src_is_sharded: bool = False,
        x_dst_is_sharded: bool = False,
        keep_x_dst_sharded: bool = True,
        **kwargs,
    ) -> PairTensor:
        x_dst = super().forward(
            x,
            batch_size,
            shard_shapes,
            model_comm_group,
            x_src_is_sharded,
            x_dst_is_sharded,
            keep_x_dst_sharded,
            **kwargs,
        )
        return x[0], x_dst


class GraphTransformerBackwardMapper(BackwardMapperPostProcessMixin, GraphTransformerBaseMapper):
    """Graph Transformer Mapper from hidden -> data."""

    def __init__(
        self,
        *,
        in_channels_src: int,
        in_channels_dst: int,
        hidden_dim: int,
        out_channels_dst: Optional[int] = None,
        trainable_size: int,
        num_chunks: int,
        num_heads: int,
        mlp_hidden_ratio: int,
        sub_graph: HeteroData,
        sub_graph_edge_attributes: list[str],
        src_grid_size: int,
        dst_grid_size: int,
        qk_norm: bool = False,
        initialise_data_extractor_zero: bool = False,
        cpu_offload: bool = False,
        layer_kernels: DotDict = None,
        shard_strategy: str = "edges",
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
        out_channels_dst : int
            Output channels of the destination node
        trainable_size : int
            Trainable tensor of edge
        num_chunks : int
            Number of chunks to split into
        num_heads: int
            Number of heads in transformer
        mlp_hidden_ratio: int
            Ratio of mlp hidden dimension to embedding dimension
        sub_graph : HeteroData
            Sub graph of the full structure
        sub_graph_edge_attributes : list[str]
            Edge attributes to use
        src_grid_size : int
            Source grid size
        dst_grid_size : int
            Destination grid size
        initialise_data_extractor_zero : bool, default False:
            Whether to initialise the data extractor to zero
        qk_norm : bool, optional
            Whether to use query and key normalization, default False
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        layer_kernels : DotDict, optional
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
        shard_strategy : str, optional
            Strategy to shard tensors, by default "edges"
        """
        super().__init__(
            in_channels_src=in_channels_src,
            in_channels_dst=in_channels_dst,
            hidden_dim=hidden_dim,
            out_channels_dst=out_channels_dst,
            trainable_size=trainable_size,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            qk_norm=qk_norm,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            sub_graph=sub_graph,
            sub_graph_edge_attributes=sub_graph_edge_attributes,
            src_grid_size=src_grid_size,
            dst_grid_size=dst_grid_size,
            layer_kernels=layer_kernels,
            shard_strategy=shard_strategy,
        )

        self.node_data_extractor = nn.Sequential(
            nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.out_channels_dst)
        )
        if initialise_data_extractor_zero:
            for module in self.node_data_extractor.modules():
                if isinstance(module, nn.Linear):
                    nn.init.constant_(module.weight, 0.0)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)

    def pre_process(self, x, shard_shapes, model_comm_group=None, x_src_is_sharded=False, x_dst_is_sharded=False):
        x_src, x_dst, shapes_src, shapes_dst = super().pre_process(
            x, shard_shapes, model_comm_group, x_src_is_sharded, x_dst_is_sharded
        )
        shapes_src = change_channels_in_shape(shapes_src, self.hidden_dim)
        if not x_dst_is_sharded:
            x_dst = shard_tensor(x_dst, 0, shapes_dst, model_comm_group)
        x_dst = self.emb_nodes_dst(x_dst)
        shapes_dst = change_channels_in_shape(shapes_dst, self.hidden_dim)
        return x_src, x_dst, shapes_src, shapes_dst


class GNNBaseMapper(GraphEdgeMixin, BaseMapper):
    """Base for Graph Neural Network Mapper from hidden -> data or data -> hidden."""

    def __init__(
        self,
        *,
        in_channels_src: int,
        in_channels_dst: int,
        hidden_dim: int,
        out_channels_dst: Optional[int] = None,
        trainable_size: int,
        num_chunks: int,
        mlp_extra_layers: int,
        sub_graph: HeteroData,
        sub_graph_edge_attributes: list[str],
        src_grid_size: int,
        dst_grid_size: int,
        cpu_offload: bool = False,
        layer_kernels: DotDict,
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
        out_channels_dst : int
            Output channels of the destination node
        trainable_size : int
            Trainable tensor of edge
        num_chunks : int
            Number of chunks to split into
        num_heads: int
            Number of heads in transformer
        mlp_hidden_ratio: int
            ratio of mlp hidden dimension to embedding dimension
        sub_graph : HeteroData
            Sub graph of the full structure
        sub_graph_edge_attributes : list[str]
            Edge attributes to use
        src_grid_size : int
            Source grid size
        dst_grid_size : int
            Destination grid size
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        layer_kernels : DotDict, optional
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
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

        self._register_edges(sub_graph, sub_graph_edge_attributes, src_grid_size, dst_grid_size, trainable_size)

        self.emb_edges = MLP(
            in_features=self.edge_dim,
            hidden_dim=hidden_dim,
            out_features=hidden_dim,
            layer_kernels=self.layer_factory,
            n_extra_layers=mlp_extra_layers,
        )

        self.trainable = TrainableTensor(trainable_size=trainable_size, tensor_size=self.edge_attr.shape[0])

    def prepare_edges(
        self,
        size: tuple[int, int],
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[Tensor, Adj]:
        edge_attr = self.trainable(self.edge_attr, batch_size)
        edge_index = self._expand_edges(self.edge_index_base, self.edge_inc, batch_size)
        edge_attr, edge_index, shapes_edge_attr, shapes_edge_idx = sort_edges_1hop_sharding(
            size, edge_attr, edge_index, model_comm_group
        )

        edge_index = shard_tensor(edge_index, 1, shapes_edge_idx, model_comm_group)
        edge_attr = shard_tensor(edge_attr, 0, shapes_edge_attr, model_comm_group)
        edge_attr = self.emb_edges(edge_attr)
        return edge_attr, edge_index

    def mapper_forward(
        self,
        x: PairTensor,
        batch_size: int,
        shard_shapes: tuple[list[list[int]], list[list[int]]],
        model_comm_group: Optional[ProcessGroup] = None,
        x_src_is_sharded: bool = False,
        x_dst_is_sharded: bool = False,
        keep_x_dst_sharded: bool = False,
        **kwargs,
    ) -> PairTensor:

        size = (sum(x[0] for x in shard_shapes[0]), sum(x[0] for x in shard_shapes[1]))

        edge_attr, edge_index = self.prepare_edges(size, batch_size, model_comm_group)

        x_src, x_dst, shapes_src, shapes_dst = self.pre_process(
            x, shard_shapes, model_comm_group, x_src_is_sharded, x_dst_is_sharded
        )

        (x_src, x_dst), edge_attr = self.proc(
            (x_src, x_dst),
            edge_attr,
            edge_index,
            (shapes_src, shapes_dst),
            model_comm_group,
            size=size,
            **kwargs,
        )

        x_dst = self.post_process(x_dst, shapes_dst, model_comm_group, keep_x_dst_sharded)

        return x_src, x_dst

    def forward(
        self,
        x: PairTensor,
        batch_size: int,
        shard_shapes: tuple[list[list[int]], list[list[int]]],
        model_comm_group: Optional[ProcessGroup] = None,
        x_src_is_sharded: bool = False,
        x_dst_is_sharded: bool = False,
        keep_x_dst_sharded: bool = False,
        **kwargs,
    ) -> PairTensor:
        return checkpoint(
            self.mapper_forward,
            x=x,
            batch_size=batch_size,
            shard_shapes=shard_shapes,
            model_comm_group=model_comm_group,
            x_src_is_sharded=x_src_is_sharded,
            x_dst_is_sharded=x_dst_is_sharded,
            keep_x_dst_sharded=keep_x_dst_sharded,
            **kwargs,
            use_reentrant=False,
        )


class GNNForwardMapper(ForwardMapperPreProcessMixin, GNNBaseMapper):
    """Graph Neural Network Mapper data -> hidden."""

    def __init__(
        self,
        *,
        in_channels_src: int,
        in_channels_dst: int,
        hidden_dim: int,
        out_channels_dst: Optional[int] = None,
        trainable_size: int,
        num_chunks: int,
        mlp_extra_layers: int,
        sub_graph: HeteroData,
        sub_graph_edge_attributes: list[str],
        src_grid_size: int,
        dst_grid_size: int,
        cpu_offload: bool = False,
        layer_kernels: DotDict,
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
        out_channels_dst : int
            Output channels of the destination node, by default None
        trainable_size : int
            Trainable tensor of edge
        num_chunks: int
            Number of chunks to split into
        mlp_extra_layers : int, optional
            Number of extra layers in MLP, by default 0
        sub_graph : HeteroData
            Sub graph of the full structure
        sub_graph_edge_attributes : list[str]
            Edge attributes to use
        src_grid_size : int
            Source grid size
        dst_grid_size : int
            Destination grid size
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
        """
        super().__init__(
            in_channels_src=in_channels_src,
            in_channels_dst=in_channels_dst,
            hidden_dim=hidden_dim,
            out_channels_dst=out_channels_dst,
            trainable_size=trainable_size,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            mlp_extra_layers=mlp_extra_layers,
            sub_graph=sub_graph,
            sub_graph_edge_attributes=sub_graph_edge_attributes,
            src_grid_size=src_grid_size,
            dst_grid_size=dst_grid_size,
            layer_kernels=layer_kernels,
        )

        self.proc = GraphConvMapperBlock(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            layer_kernels=self.layer_factory,
            mlp_extra_layers=mlp_extra_layers,
            update_src_nodes=True,
            num_chunks=num_chunks,
        )

        self.offload_layers(cpu_offload)

        self.emb_nodes_src = MLP(
            in_features=in_channels_src,
            hidden_dim=hidden_dim,
            out_features=hidden_dim,
            layer_kernels=self.layer_factory,
            n_extra_layers=mlp_extra_layers,
        )

        self.emb_nodes_dst = MLP(
            in_features=in_channels_dst,
            hidden_dim=hidden_dim,
            out_features=hidden_dim,
            layer_kernels=self.layer_factory,
            n_extra_layers=mlp_extra_layers,
        )


class GNNBackwardMapper(BackwardMapperPostProcessMixin, GNNBaseMapper):
    """Graph Neural Network Mapper from hidden -> data."""

    def __init__(
        self,
        *,
        in_channels_src: int,
        in_channels_dst: int,
        hidden_dim: int,
        out_channels_dst: Optional[int] = None,
        trainable_size: int,
        num_chunks: int,
        mlp_extra_layers: int,
        sub_graph: HeteroData,
        sub_graph_edge_attributes: list[str],
        src_grid_size: int,
        dst_grid_size: int,
        cpu_offload: bool = False,
        layer_kernels: DotDict,
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
        out_channels_dst : int
            Output channels of the destination node
        trainable_size : int
            Trainable tensor of edge
        num_chunks: int
            Number of chunks to split into
        mlp_extra_layers : int
            Number of extra layers in MLP
        sub_graph : HeteroData
            Sub graph of the full structure
        sub_graph_edge_attributes : list[str]
            Edge attributes to use
        src_grid_size : int
            Source grid size
        dst_grid_size : int
            Destination grid size
        cpu_offload : bool
            Whether to offload processing to CPU, by default False
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
        """
        super().__init__(
            in_channels_src=in_channels_src,
            in_channels_dst=in_channels_dst,
            hidden_dim=hidden_dim,
            trainable_size=trainable_size,
            out_channels_dst=out_channels_dst,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            mlp_extra_layers=mlp_extra_layers,
            sub_graph=sub_graph,
            sub_graph_edge_attributes=sub_graph_edge_attributes,
            src_grid_size=src_grid_size,
            dst_grid_size=dst_grid_size,
            layer_kernels=layer_kernels,
        )

        self.proc = GraphConvMapperBlock(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            layer_kernels=self.layer_factory,
            mlp_extra_layers=mlp_extra_layers,
            update_src_nodes=False,
            num_chunks=num_chunks,
        )

        self.offload_layers(cpu_offload)

        self.node_data_extractor = MLP(
            in_features=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            out_features=self.out_channels_dst,
            layer_kernels=self.layer_factory,
            n_extra_layers=mlp_extra_layers,
            layer_norm=False,
            final_activation=False,
        )

    def pre_process(self, x, shard_shapes, model_comm_group=None, x_src_is_sharded=False, x_dst_is_sharded=False):
        x_src, x_dst, shapes_src, shapes_dst = super().pre_process(
            x, shard_shapes, model_comm_group, x_src_is_sharded, x_dst_is_sharded
        )
        shapes_src = change_channels_in_shape(shapes_src, self.hidden_dim)
        shapes_dst = change_channels_in_shape(shapes_dst, self.hidden_dim)
        return x_src, x_dst, shapes_src, shapes_dst

    def forward(
        self,
        x: PairTensor,
        batch_size: int,
        shard_shapes: tuple[list[list[int]], list[list[int]]],
        model_comm_group: Optional[ProcessGroup] = None,
        x_src_is_sharded: bool = False,
        x_dst_is_sharded: bool = False,
        keep_x_dst_sharded: bool = False,
        **kwargs,
    ) -> Tensor:

        _, x_dst = super().forward(
            x,
            batch_size,
            shard_shapes,
            model_comm_group,
            x_src_is_sharded,
            x_dst_is_sharded,
            keep_x_dst_sharded,
            **kwargs,
        )
        return x_dst


class TransformerBaseMapper(BaseMapper):
    """Transformer Base Mapper from hidden -> data or data -> hidden."""

    def __init__(
        self,
        *,
        in_channels_src: int,
        in_channels_dst: int,
        hidden_dim: int,
        out_channels_dst: Optional[int] = None,
        num_chunks: int,
        num_heads: int,
        mlp_hidden_ratio: int,
        window_size: Optional[int] = None,
        dropout_p: float = 0.0,
        qk_norm: bool = False,
        attention_implementation: str = "flash_attention",
        softcap: Optional[float] = None,
        use_alibi_slopes: bool = False,
        use_rotary_embeddings: bool = False,
        cpu_offload: bool = False,
        layer_kernels: DotDict,
    ) -> None:
        """Initialize TransformerBaseMapper.

        Parameters
        ----------
        in_channels_src : int
            Input channels of the source node
        in_channels_dst : int
            Input channels of the destination node
        hidden_dim : int
            Hidden dimension
        out_channels_dst : int, optional
            Output channels of the destination node, by default None
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
            in_channels_src=in_channels_src,
            in_channels_dst=in_channels_dst,
            hidden_dim=hidden_dim,
            out_channels_dst=out_channels_dst,
            num_chunks=num_chunks,
            layer_kernels=layer_kernels,
            cpu_offload=cpu_offload,
        )

        self.proc = TransformerMapperBlock(
            num_channels=hidden_dim,
            hidden_dim=mlp_hidden_ratio * hidden_dim,
            num_heads=num_heads,
            window_size=window_size,
            layer_kernels=self.layer_factory,
            dropout_p=dropout_p,
            qk_norm=qk_norm,
            attention_implementation=attention_implementation,
            softcap=softcap,
            use_alibi_slopes=use_alibi_slopes,
            use_rotary_embeddings=use_rotary_embeddings,
        )

        self.offload_layers(cpu_offload)

        self.emb_nodes_dst = nn.Linear(self.in_channels_dst, self.hidden_dim)

    def mapper_forward(
        self,
        x: PairTensor,
        batch_size: int,
        shard_shapes: tuple[list[list[int]], list[list[int]]],
        model_comm_group: Optional[ProcessGroup] = None,
        x_src_is_sharded: bool = False,
        x_dst_is_sharded: bool = False,
        keep_x_dst_sharded: bool = False,
    ) -> PairTensor:

        x_src, x_dst, shapes_src, shapes_dst = self.pre_process(
            x, shard_shapes, model_comm_group, x_src_is_sharded, x_dst_is_sharded
        )

        (x_src, x_dst), _ = self.proc(
            (x_src, x_dst),
            (shapes_src, shapes_dst),
            batch_size,
            model_comm_group,
        )

        x_dst = self.post_process(x_dst, shapes_dst, model_comm_group, keep_x_dst_sharded=keep_x_dst_sharded)

        return x_dst

    def forward(
        self,
        x: PairTensor,
        batch_size: int,
        shard_shapes: tuple[list[list[int]], list[list[int]]],
        model_comm_group: Optional[ProcessGroup] = None,
        x_src_is_sharded: bool = False,
        x_dst_is_sharded: bool = False,
        keep_x_dst_sharded: bool = False,
        **kwargs,
    ) -> PairTensor:
        return checkpoint(
            self.mapper_forward,
            x=x,
            batch_size=batch_size,
            shard_shapes=shard_shapes,
            model_comm_group=model_comm_group,
            x_src_is_sharded=x_src_is_sharded,
            x_dst_is_sharded=x_dst_is_sharded,
            keep_x_dst_sharded=keep_x_dst_sharded,
            **kwargs,
            use_reentrant=False,
        )


class TransformerForwardMapper(ForwardMapperPreProcessMixin, TransformerBaseMapper):
    """Transformer Mapper from data -> hidden."""

    def __init__(
        self,
        *,
        in_channels_src: int,
        in_channels_dst: int,
        hidden_dim: int,
        out_channels_dst: Optional[int] = None,
        num_chunks: int,
        num_heads: int,
        mlp_hidden_ratio: int,
        qk_norm: bool = False,
        dropout_p: float = 0.0,
        attention_implementation: str = "flash_attention",
        softcap: float = None,
        use_alibi_slopes: bool = False,
        cpu_offload: bool = False,
        window_size: Optional[int] = None,
        use_rotary_embeddings: bool = False,
        layer_kernels: DotDict,
        **kwargs,  # accept not needed extra arguments like subgraph etc.
    ) -> None:
        """Initialize TransformerForwardMapper.

        Parameters
        ----------
        in_channels_src : int
            Input channels of the source node
        in_channels_dst : int
            Input channels of the destination node
        hidden_dim : int
            Hidden dimension
        out_channels_dst : int, optional
            Output channels of the destination node, by default None
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
            in_channels_src=in_channels_src,
            in_channels_dst=in_channels_dst,
            hidden_dim=hidden_dim,
            layer_kernels=layer_kernels,
            out_channels_dst=out_channels_dst,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            window_size=window_size,
            dropout_p=dropout_p,
            qk_norm=qk_norm,
            attention_implementation=attention_implementation,
            softcap=softcap,
            use_alibi_slopes=use_alibi_slopes,
            use_rotary_embeddings=use_rotary_embeddings,
        )

        self.emb_nodes_src = nn.Linear(self.in_channels_src, self.hidden_dim)

    def forward(
        self,
        x: PairTensor,
        batch_size: int,
        shard_shapes: tuple[list[list[int]], list[list[int]], list[list[int]]],
        model_comm_group: Optional[ProcessGroup] = None,
        x_src_is_sharded: bool = False,
        x_dst_is_sharded: bool = False,
        keep_x_dst_sharded: bool = False,
    ) -> PairTensor:
        x_dst = super().forward(
            x, batch_size, shard_shapes, model_comm_group, x_src_is_sharded, x_dst_is_sharded, keep_x_dst_sharded
        )
        return x[0], x_dst


class TransformerBackwardMapper(BackwardMapperPostProcessMixin, TransformerBaseMapper):
    """Graph Transformer Mapper from hidden -> data."""

    def __init__(
        self,
        *,
        in_channels_src: int,
        in_channels_dst: int,
        hidden_dim: int,
        out_channels_dst: Optional[int] = None,
        num_chunks: int,
        num_heads: int,
        mlp_hidden_ratio: int,
        qk_norm: bool = False,
        dropout_p: float = 0.0,
        attention_implementation: str = "flash_attention",
        softcap: float = None,
        use_alibi_slopes: bool = False,
        cpu_offload: bool = False,
        window_size: Optional[int] = None,
        use_rotary_embeddings: bool = False,
        layer_kernels: DotDict,
        **kwargs,  # accept not needed extra arguments like subgraph etc.
    ) -> None:
        """Initialize TransformerBackwardMapper.

        Parameters
        ----------
        in_channels_src : int
            Input channels of the source node
        in_channels_dst : int
            Input channels of the destination node
        hidden_dim : int
            Hidden dimension
        out_channels_dst : int, optional
            Output channels of the destination node, by default None
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
            in_channels_src=in_channels_src,
            in_channels_dst=in_channels_dst,
            hidden_dim=hidden_dim,
            layer_kernels=layer_kernels,
            out_channels_dst=out_channels_dst,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            window_size=window_size,
            dropout_p=dropout_p,
            qk_norm=qk_norm,
            attention_implementation=attention_implementation,
            softcap=softcap,
            use_alibi_slopes=use_alibi_slopes,
            use_rotary_embeddings=use_rotary_embeddings,
        )

        self.node_data_extractor = nn.Sequential(
            nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.out_channels_dst)
        )

    def pre_process(self, x, shard_shapes, model_comm_group=None, x_src_is_sharded=False, x_dst_is_sharded=False):
        x_src, x_dst, shapes_src, shapes_dst = super().pre_process(
            x, shard_shapes, model_comm_group, x_src_is_sharded, x_dst_is_sharded
        )
        shapes_src = change_channels_in_shape(shapes_src, self.hidden_dim)
        if not x_dst_is_sharded:
            x_dst = shard_tensor(x_dst, 0, shapes_dst, model_comm_group)
        x_dst = self.emb_nodes_dst(x_dst)
        shapes_dst = change_channels_in_shape(shapes_dst, self.hidden_dim)
        return x_src, x_dst, shapes_src, shapes_dst
