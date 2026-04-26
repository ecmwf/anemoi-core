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
from abc import abstractmethod
from typing import Optional

import torch
from torch import Tensor
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.typing import Adj
from torch_geometric.typing import PairTensor

from anemoi.models.distributed.graph import ensure_sharded
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.khop_edges import GraphPartition
from anemoi.models.distributed.khop_edges import build_graph_partition
from anemoi.models.distributed.khop_edges import build_graph_partition_from_shard_info
from anemoi.models.distributed.khop_edges import shard_edges_1hop
from anemoi.models.distributed.khop_edges import shard_graph_to_local
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.layers.block import GraphConvMapperBlock
from anemoi.models.layers.block import GraphTransformerMapperBlock
from anemoi.models.layers.block import TransformerMapperBlock
from anemoi.models.layers.mlp import MLP
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.models.layers.utils import maybe_checkpoint
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)

# Number of chunks used in inference (https://github.com/ecmwf/anemoi-core/pull/406)
NUM_CHUNKS_INFERENCE = int(os.environ.get("ANEMOI_INFERENCE_NUM_CHUNKS", "1"))
NUM_CHUNKS_INFERENCE_MAPPER = int(os.environ.get("ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER", NUM_CHUNKS_INFERENCE))


class BaseMapper(nn.Module, ABC):
    """Base Mapper from source dimension to destination dimension.

    Subclasses must implement pre_process() and post_process() methods
    specialized for their mapper type.
    """

    def __init__(
        self,
        *,
        in_channels_src: int,
        in_channels_dst: int,
        hidden_dim: int,
        out_channels_dst: Optional[int] = None,
        cpu_offload: bool = False,
        gradient_checkpointing: bool = True,
        layer_kernels: DotDict,
        **kwargs,
    ) -> None:
        """Initialize BaseMapper."""
        super().__init__()

        self.in_channels_src = in_channels_src
        self.in_channels_dst = in_channels_dst
        self.hidden_dim = hidden_dim
        self.out_channels_dst = out_channels_dst
        self.gradient_checkpointing = gradient_checkpointing
        self.layer_factory = load_layer_kernels(layer_kernels)
        self.activation = self.layer_factory.Activation()

        self.proc = NotImplemented

    def offload_layers(self, cpu_offload):
        if cpu_offload:
            self.proc = nn.ModuleList([offload_wrapper(x) for x in self.proc])

    @abstractmethod
    def pre_process(self, x):
        pass

    @abstractmethod
    def post_process(self, x_dst):
        pass

    @abstractmethod
    def forward(
        self,
        x: PairTensor,
        batch_size: int,
        shard_info: BipartiteGraphShardInfo,
        edge_attr: Optional[Tensor] = None,
        edge_index: Optional[Adj] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        keep_x_dst_sharded: bool = False,
        **kwargs,
    ):
        """Forward pass of the mapper.

        Parameters
        ----------
        x : PairTensor
            Input tensor pair (source, destination)
        batch_size : int
            Batch size
        shard_info : BipartiteGraphShardInfo
            Shard metadata. Each field is a list of per-rank partition sizes
            along the sharded dimension, or None if the tensor is replicated.
        edge_attr : Tensor, optional
            Edge attributes (required for graph-based mappers)
        edge_index : Adj, optional
            Edge indices (required for graph-based mappers)
        model_comm_group : ProcessGroup, optional
            Model communication group
        keep_x_dst_sharded : bool, optional
            Whether to keep destination sharded, by default False
        """
        pass


class GraphTransformerBaseMapper(BaseMapper, ABC):
    """Graph Transformer Base Mapper from hidden -> data or data -> hidden."""

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
        edge_dim: int,
        attn_channels: Optional[int] = None,
        qk_norm: bool = False,
        cpu_offload: bool = False,
        layer_kernels: DotDict = None,
        shard_strategy: str = "edges",
        graph_attention_backend: str = "triton",
        edge_pre_mlp: bool = False,
        **kwargs,
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
        num_chunks : int
            Number of chunks to split into
        num_heads: int
            Number of heads in transformer
        mlp_hidden_ratio: int
            ratio of mlp hidden dimension to embedding dimension
        edge_dim : int
            Edge feature dimension
        attn_channels : int, optional
            Internal attention width used for q/k/v and edge projections. If
            None, defaults to the hidden dimension. This allows reducing the
            number of channels used for the attention computation without
            changing the width of the surrounding MLPs.
        qk_norm : bool, optional
            Whether to use query and key normalization, default False
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        layer_kernels : DotDict, optional
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
        shard_strategy : str, optional
            Strategy to shard tensors, by default "edges"
        graph_attention_backend: str, by default "triton"
            Backend to use for graph transformer conv, options are "triton" and "pyg"
        edge_pre_mlp: bool, by default False
            Allow for edge feature mixing
        """
        super().__init__(
            in_channels_src=in_channels_src,
            in_channels_dst=in_channels_dst,
            hidden_dim=hidden_dim,
            out_channels_dst=out_channels_dst,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            layer_kernels=layer_kernels,
            **kwargs,
        )

        self.num_chunks = num_chunks

        Linear = self.layer_factory.Linear

        self.proc = GraphTransformerMapperBlock(
            in_channels=hidden_dim,
            hidden_dim=mlp_hidden_ratio * hidden_dim,
            out_channels=hidden_dim,
            attn_channels=attn_channels,
            num_heads=num_heads,
            edge_dim=edge_dim,
            qk_norm=qk_norm,
            layer_kernels=self.layer_factory,
            shard_strategy=shard_strategy,
            graph_attention_backend=graph_attention_backend,
            edge_pre_mlp=edge_pre_mlp,
        )

        self.offload_layers(cpu_offload)

        self.emb_nodes_dst = Linear(self.in_channels_dst, self.hidden_dim)

        self.shard_strategy = shard_strategy

        assert shard_strategy in ["heads", "edges"], (
            f"Invalid shard strategy '{shard_strategy}' for {self.__class__.__name__}. "
            f"Supported strategies are 'heads' and 'edges'."
        )

    def prepare_edge_sharding_wrapper(
        self,
        x: PairTensor,
        shard_info: BipartiteGraphShardInfo,
        batch_size: int,
        edge_attr: Tensor,
        edge_index: Adj,
        model_comm_group: Optional[ProcessGroup] = None,
        cond: Optional[tuple[Tensor, Tensor]] = None,
    ):
        # build a GraphPartition for the distributed shard (across GPUs)
        shard_partition = build_graph_partition_from_shard_info(
            edge_index,
            x,
            shard_info,
            model_comm_group,
        )

        # shard to local rank: gathers src, shards dst+edges, relabels dst, drops unconnected src
        (x_src, x_dst), edge_attr, edge_index, shard_info, cond = shard_graph_to_local(
            shard_partition,
            x,
            edge_attr,
            edge_index,
            shard_info,
            model_comm_group,
            cond=cond,
        )

        # build a second GraphPartition for local chunking within this shard
        num_chunks = max(self.num_chunks, NUM_CHUNKS_INFERENCE_MAPPER)
        chunk_partition = build_graph_partition(
            edge_index,
            num_parts=num_chunks,
            num_nodes=(x_src.shape[0], x_dst.shape[0]),
        )

        return x_src, x_dst, edge_attr, edge_index, shard_info, cond, chunk_partition

    def run_processor_chunk(
        self,
        chunk_partition: GraphPartition,
        chunk_id: int,
        x: tuple[Tensor, Tensor],
        edge_attr: Tensor,
        edge_index: Adj,
        shard_info: BipartiteGraphShardInfo,
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
        cond: Optional[tuple[Tensor, Tensor]] = None,
        **kwargs,
    ) -> Tensor:
        # O(1) slicing: extract subgraph for this chunk
        (x_src_chunk, x_dst_chunk), edge_attr_chunk, edge_index_chunk, cond_chunk = chunk_partition.materialise(
            chunk_id, x, edge_attr, edge_index, cond=cond
        )
        chunk_size = (x_src_chunk.shape[0], x_dst_chunk.shape[0])

        # pre-process chunk, embedding x_src/x_dst
        x_src_chunk, x_dst_chunk = self.pre_process((x_src_chunk, x_dst_chunk))

        (_, x_dst_out), _ = self.proc(
            (x_src_chunk, x_dst_chunk),
            edge_attr_chunk,
            edge_index_chunk,
            shard_info,
            batch_size,
            chunk_size,
            model_comm_group,
            cond=cond_chunk,
            **kwargs,
        )

        return self.post_process(x_dst_out)

    def mapper_forward_with_edge_sharding(
        self,
        x: PairTensor,
        batch_size: int,
        shard_info: BipartiteGraphShardInfo,
        edge_attr: Tensor,
        edge_index: Adj,
        model_comm_group: Optional[ProcessGroup] = None,
        keep_x_dst_sharded: bool = False,
        cond: Optional[tuple[Tensor, Tensor]] = None,
        **kwargs,
    ) -> PairTensor:
        x_src, x_dst, edge_attr, edge_index, shard_info, cond, chunk_partition = maybe_checkpoint(
            self.prepare_edge_sharding_wrapper,
            self.gradient_checkpointing,
            x,
            shard_info,
            batch_size,
            edge_attr,
            edge_index,
            model_comm_group,
            cond,
        )

        out_channels = self.out_channels_dst if self.out_channels_dst is not None else self.hidden_dim
        out_type = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else x_dst.dtype
        out_dst = torch.empty((*x_dst.shape[:-1], out_channels), device=x_dst.device, dtype=out_type)

        for chunk_id in range(chunk_partition.num_parts):
            dst_range = chunk_partition._get_dst_range(chunk_id)
            out_dst[dst_range] = maybe_checkpoint(
                self.run_processor_chunk,
                self.gradient_checkpointing,
                chunk_partition,
                chunk_id,
                (x_src, x_dst),
                edge_attr,
                edge_index,
                shard_info,
                batch_size,
                model_comm_group,
                cond,
                **kwargs,
            ).to(dtype=out_type)

        if not keep_x_dst_sharded:  # gather after processing chunks
            out_dst = gather_tensor(out_dst, 0, shard_info.dst_nodes, model_comm_group)

        return out_dst

    def mapper_forward_with_heads_sharding(
        self,
        x: PairTensor,
        batch_size: int,
        shard_info: BipartiteGraphShardInfo,
        edge_attr: Tensor,
        edge_index: Adj,
        model_comm_group: Optional[ProcessGroup] = None,
        keep_x_dst_sharded: bool = False,
        **kwargs,
    ) -> PairTensor:
        x_src, x_dst = x
        shard_sizes_src, shard_sizes_dst, shard_sizes_edges = (
            shard_info.src_nodes,
            shard_info.dst_nodes,
            shard_info.edges,
        )

        if shard_info.edges_are_sharded():
            # Heads sharding needs full edge_index — gather it
            edge_index = gather_tensor(edge_index, 1, shard_sizes_edges, model_comm_group)

        # ensure everything is sharded for all-all later
        x_src, shard_sizes_src = ensure_sharded(x_src, 0, shard_sizes_src, model_comm_group)
        x_dst, shard_sizes_dst = ensure_sharded(x_dst, 0, shard_sizes_dst, model_comm_group)
        edge_attr, shard_sizes_edges = ensure_sharded(edge_attr, 0, shard_sizes_edges, model_comm_group)
        size = (sum(shard_sizes_src), sum(shard_sizes_dst))

        # update ShardInfo
        shard_info = BipartiteGraphShardInfo(
            src_nodes=shard_sizes_src,
            dst_nodes=shard_sizes_dst,
            edges=shard_sizes_edges,
        )

        x_src, x_dst = self.pre_process((x_src, x_dst))

        (x_src, x_dst), edge_attr = self.proc(
            x=(x_src, x_dst),
            edge_attr=edge_attr,
            edge_index=edge_index,
            shard_info=shard_info,
            batch_size=batch_size,
            size=size,
            model_comm_group=model_comm_group,
            **kwargs,
        )

        x_dst = self.post_process(x_dst)

        if not keep_x_dst_sharded:  # gather after processing
            x_dst = gather_tensor(x_dst, 0, shard_info.dst_nodes, model_comm_group)

        return x_dst

    def forward(
        self,
        x: PairTensor,
        batch_size: int,
        shard_info: BipartiteGraphShardInfo,
        edge_attr: Tensor,
        edge_index: Adj,
        model_comm_group: Optional[ProcessGroup] = None,
        keep_x_dst_sharded: bool = False,
        **kwargs,
    ) -> PairTensor:

        kwargs_forward = {
            "x": x,
            "batch_size": batch_size,
            "shard_info": shard_info,
            "edge_attr": edge_attr,
            "edge_index": edge_index,
            "model_comm_group": model_comm_group,
            "keep_x_dst_sharded": keep_x_dst_sharded,
            **kwargs,
        }

        if self.shard_strategy == "edges" and model_comm_group is not None:
            return self.mapper_forward_with_edge_sharding(**kwargs_forward)
        else:  # self.shard_strategy == "heads" or single-GPU fallback
            return maybe_checkpoint(
                self.mapper_forward_with_heads_sharding,
                self.gradient_checkpointing,
                **kwargs_forward,
            )


class GraphTransformerForwardMapper(GraphTransformerBaseMapper):
    """Graph Transformer Mapper from data -> hidden."""

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
        edge_dim: int,
        attn_channels: Optional[int] = None,
        qk_norm: bool = False,
        cpu_offload: bool = False,
        layer_kernels: DotDict = None,
        shard_strategy: str = "edges",
        graph_attention_backend: str = "triton",
        edge_pre_mlp: bool = False,
        **kwargs,
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
        out_channels_dst : int, optional
            Must remain ``None`` for forward graph-transformer mappers.
        num_chunks : int
            Number of chunks to split into
        num_heads: int
            Number of heads in transformer
        mlp_hidden_ratio: int
            ratio of mlp hidden dimension to embedding dimension
        edge_dim : int
            Edge feature dimension
        attn_channels : int, optional
            Internal attention width used for q/k/v and edge projections. If
            None, defaults to the hidden dimension. This allows reducing the
            number of channels used for the attention computation without
            changing the width of the surrounding MLPs.
        qk_norm : bool, optional
            Whether to use query and key normalization, default False
        cpu_offload : bool
            Whether to offload processing to CPU, by default False
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
        shard_strategy : str, optional
            Strategy to shard tensors, by default "edges"
        graph_attention_backend: str, by default "triton"
            Backend to use for graph transformer conv, options are "triton" and "pyg"
        edge_pre_mlp: bool, by default False
            Allow for edge feature mixing
        """
        assert out_channels_dst is None, "GraphTransformerForwardMapper does not support out_channels_dst."
        super().__init__(
            in_channels_src=in_channels_src,
            in_channels_dst=in_channels_dst,
            hidden_dim=hidden_dim,
            out_channels_dst=None,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            qk_norm=qk_norm,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            edge_dim=edge_dim,
            attn_channels=attn_channels,
            layer_kernels=layer_kernels,
            shard_strategy=shard_strategy,
            graph_attention_backend=graph_attention_backend,
            edge_pre_mlp=edge_pre_mlp,
            **kwargs,
        )

        self.emb_nodes_src = self.layer_factory.Linear(self.in_channels_src, self.hidden_dim)

    def pre_process(self, x):
        x_src, x_dst = x
        x_src = self.emb_nodes_src(x_src)
        x_dst = self.emb_nodes_dst(x_dst)
        return x_src, x_dst

    def post_process(self, x_dst, **kwargs):
        return x_dst

    def forward(
        self,
        x: PairTensor,
        batch_size: int,
        shard_info: BipartiteGraphShardInfo,
        edge_attr: Tensor,
        edge_index: Adj,
        model_comm_group: Optional[ProcessGroup] = None,
        keep_x_dst_sharded: bool = True,
        **kwargs,
    ) -> PairTensor:
        x_dst = super().forward(
            x,
            batch_size,
            shard_info,
            edge_attr,
            edge_index,
            model_comm_group,
            keep_x_dst_sharded,
            **kwargs,
        )
        return x[0], x_dst


class GraphTransformerBackwardMapper(GraphTransformerBaseMapper):
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
        edge_dim: int,
        attn_channels: Optional[int] = None,
        qk_norm: bool = False,
        initialise_data_extractor_zero: bool = False,
        cpu_offload: bool = False,
        layer_kernels: DotDict = None,
        shard_strategy: str = "edges",
        graph_attention_backend: str = "triton",
        edge_pre_mlp: bool = False,
        **kwargs,
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
        num_chunks : int
            Number of chunks to split into
        num_heads: int
            Number of heads in transformer
        mlp_hidden_ratio: int
            Ratio of mlp hidden dimension to embedding dimension
        edge_dim : int
            Edge feature dimension
        attn_channels : int, optional
            Internal attention width used for q/k/v and edge projections. If
            None, defaults to the hidden dimension. This allows reducing the
            number of channels used for the attention computation without
            changing the width of the surrounding MLPs.
        qk_norm : bool, optional
            Whether to use query and key normalization, default False
        initialise_data_extractor_zero : bool, default False:
            Whether to initialise the data extractor to zero
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        layer_kernels : DotDict, optional
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
        shard_strategy : str, optional
            Strategy to shard tensors, by default "edges"
        graph_attention_backend: str, by default "triton"
            Backend to use for graph transformer conv, options are "triton" and "pyg"
        edge_pre_mlp: bool, by default False
            Allow for edge feature mixing
        """
        super().__init__(
            in_channels_src=in_channels_src,
            in_channels_dst=in_channels_dst,
            hidden_dim=hidden_dim,
            out_channels_dst=out_channels_dst,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            qk_norm=qk_norm,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            edge_dim=edge_dim,
            attn_channels=attn_channels,
            layer_kernels=layer_kernels,
            shard_strategy=shard_strategy,
            graph_attention_backend=graph_attention_backend,
            edge_pre_mlp=edge_pre_mlp,
            **kwargs,
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

    def pre_process(self, x):
        x_src, x_dst = x
        x_dst = self.emb_nodes_dst(x_dst)
        return x_src, x_dst

    def post_process(self, x_dst):
        return self.node_data_extractor(x_dst)


class GNNBaseMapper(BaseMapper, ABC):
    """Base for Graph Neural Network Mapper from hidden -> data or data -> hidden."""

    def __init__(
        self,
        *,
        in_channels_src: int,
        in_channels_dst: int,
        hidden_dim: int,
        out_channels_dst: Optional[int] = None,
        num_chunks: int,
        mlp_extra_layers: int,
        edge_dim: int,
        cpu_offload: bool = False,
        layer_kernels: DotDict = None,
        **kwargs,
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
        out_channels_dst : int, optional
            Output channels of the destination node
        num_chunks : int
            Number of chunks to split into
        mlp_extra_layers : int
            Number of extra layers in MLP
        edge_dim : int
            Edge feature dimension
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
            **kwargs,
        )

        self.emb_edges = MLP(
            in_features=edge_dim,
            hidden_dim=hidden_dim,
            out_features=hidden_dim,
            layer_kernels=self.layer_factory,
            n_extra_layers=mlp_extra_layers,
        )

    def mapper_forward(
        self,
        x: PairTensor,
        batch_size: int,
        shard_info: BipartiteGraphShardInfo,
        edge_attr: Tensor,
        edge_index: Adj,
        model_comm_group: Optional[ProcessGroup] = None,
        keep_x_dst_sharded: bool = False,
        **kwargs,
    ) -> PairTensor:
        x_src, x_dst = x
        shard_sizes_src, shard_sizes_dst, shard_sizes_edges = (
            shard_info.src_nodes,
            shard_info.dst_nodes,
            shard_info.edges,
        )

        # Ensure src and dst are sharded
        x_src, shard_sizes_src = ensure_sharded(x_src, 0, shard_sizes_src, model_comm_group)
        x_dst, shard_sizes_dst = ensure_sharded(x_dst, 0, shard_sizes_dst, model_comm_group)
        size = (sum(shard_sizes_src), sum(shard_sizes_dst))

        if not shard_info.edges_are_sharded():
            # Edges not pre-sharded, do 1-hop sorting and sharding here
            edge_attr, edge_index, shard_sizes_edges = shard_edges_1hop(
                edge_attr, edge_index, size[0], size[1], model_comm_group
            )

        shard_info = BipartiteGraphShardInfo(
            src_nodes=shard_sizes_src,
            dst_nodes=shard_sizes_dst,
            edges=shard_sizes_edges,
        )

        edge_attr = self.emb_edges(edge_attr)

        x_src, x_dst = self.pre_process((x_src, x_dst))

        (x_src, x_dst), edge_attr = self.proc(
            (x_src, x_dst),
            edge_attr,
            edge_index,
            shard_info,
            model_comm_group,
            size=size,
            **kwargs,
        )

        x_dst = self.post_process(x_dst)

        if not keep_x_dst_sharded:
            x_dst = gather_tensor(x_dst, 0, shard_info.dst_nodes, model_comm_group)

        return x_src, x_dst

    def forward(
        self,
        x: PairTensor,
        batch_size: int,
        shard_info: BipartiteGraphShardInfo,
        edge_attr: Tensor,
        edge_index: Adj,
        model_comm_group: Optional[ProcessGroup] = None,
        keep_x_dst_sharded: bool = False,
        **kwargs,
    ) -> PairTensor:
        return maybe_checkpoint(
            self.mapper_forward,
            self.gradient_checkpointing,
            x=x,
            batch_size=batch_size,
            shard_info=shard_info,
            edge_attr=edge_attr,
            edge_index=edge_index,
            model_comm_group=model_comm_group,
            keep_x_dst_sharded=keep_x_dst_sharded,
            **kwargs,
        )


class GNNForwardMapper(GNNBaseMapper):
    """Graph Neural Network Mapper data -> hidden."""

    def __init__(
        self,
        *,
        in_channels_src: int,
        in_channels_dst: int,
        hidden_dim: int,
        out_channels_dst: Optional[int] = None,
        num_chunks: int,
        mlp_extra_layers: int,
        edge_dim: int,
        cpu_offload: bool = False,
        layer_kernels: DotDict,
        **kwargs,
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
        num_chunks: int
            Number of chunks to split into
        mlp_extra_layers : int
            Number of extra layers in MLP
        edge_dim : int
            Edge feature dimension
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
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            mlp_extra_layers=mlp_extra_layers,
            edge_dim=edge_dim,
            layer_kernels=layer_kernels,
            **kwargs,
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

    def pre_process(self, x):
        x_src, x_dst = x
        x_src = self.emb_nodes_src(x_src)
        x_dst = self.emb_nodes_dst(x_dst)
        return x_src, x_dst

    def post_process(self, x_dst, **kwargs):
        return x_dst


class GNNBackwardMapper(GNNBaseMapper):
    """Graph Neural Network Mapper from hidden -> data."""

    def __init__(
        self,
        *,
        in_channels_src: int,
        in_channels_dst: int,
        hidden_dim: int,
        out_channels_dst: Optional[int] = None,
        num_chunks: int,
        mlp_extra_layers: int,
        edge_dim: int,
        cpu_offload: bool = False,
        layer_kernels: DotDict,
        **kwargs,
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
        num_chunks: int
            Number of chunks to split into
        mlp_extra_layers : int
            Number of extra layers in MLP
        edge_dim : int
            Edge feature dimension
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
            cpu_offload=cpu_offload,
            mlp_extra_layers=mlp_extra_layers,
            edge_dim=edge_dim,
            layer_kernels=layer_kernels,
            **kwargs,
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

    def pre_process(self, x):
        x_src, x_dst = x
        return x_src, x_dst

    def post_process(self, x_dst):
        return self.node_data_extractor(x_dst)

    def forward(
        self,
        x: PairTensor,
        batch_size: int,
        shard_info: BipartiteGraphShardInfo,
        edge_attr: Tensor,
        edge_index: Adj,
        model_comm_group: Optional[ProcessGroup] = None,
        keep_x_dst_sharded: bool = False,
        **kwargs,
    ) -> Tensor:

        _, x_dst = super().forward(
            x,
            batch_size,
            shard_info,
            edge_attr,
            edge_index,
            model_comm_group,
            keep_x_dst_sharded,
            **kwargs,
        )
        return x_dst


class PointWiseMapper(BaseMapper, ABC):
    """PointWise Mapper from hidden -> data or data -> hidden."""

    def __init__(
        self,
        *,
        in_channels_src: int,
        in_channels_dst: int,
        hidden_dim: int,
        cpu_offload: bool = False,
        gradient_checkpointing: bool = True,
        layer_kernels: dict | None = None,
    ):
        super().__init__(
            in_channels_src=in_channels_src,
            in_channels_dst=in_channels_dst,
            hidden_dim=hidden_dim,
            cpu_offload=cpu_offload,
            gradient_checkpointing=gradient_checkpointing,
            layer_kernels=layer_kernels,
        )

    def mapper_forward(
        self,
        x: PairTensor,
        batch_size: int,
        shard_info: BipartiteGraphShardInfo,
        model_comm_group: Optional[ProcessGroup] = None,
        keep_x_dst_sharded: bool = False,
    ) -> PairTensor:
        x_src, x_dst = x

        # Ensure src is sharded
        x_src, shard_sizes_src = ensure_sharded(x_src, 0, shard_info.src_nodes, model_comm_group)

        x_dst = self.pre_process((x_src, x_dst))

        x_dst = self.post_process(x_dst)

        if not keep_x_dst_sharded:
            x_dst = gather_tensor(x_dst, 0, shard_sizes_src, model_comm_group)

        return x_dst

    def forward(
        self,
        x: PairTensor,
        batch_size: int,
        shard_info: BipartiteGraphShardInfo,
        edge_attr: Optional[Tensor] = None,
        edge_index: Optional[Adj] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        keep_x_dst_sharded: bool = False,
        **kwargs,
    ) -> PairTensor:
        return maybe_checkpoint(
            self.mapper_forward,
            self.gradient_checkpointing,
            x=x,
            batch_size=batch_size,
            shard_info=shard_info,
            model_comm_group=model_comm_group,
            keep_x_dst_sharded=keep_x_dst_sharded,
        )


class PointWiseForwardMapper(PointWiseMapper):
    """PointWise Mapper from data -> hidden."""

    def __init__(
        self,
        *,
        in_channels_src: int,
        in_channels_dst: int,
        hidden_dim: int,
        cpu_offload: bool = False,
        gradient_checkpointing: bool = True,
        layer_kernels: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            in_channels_src=in_channels_src,
            in_channels_dst=in_channels_dst,
            hidden_dim=hidden_dim,
            cpu_offload=cpu_offload,
            gradient_checkpointing=gradient_checkpointing,
            layer_kernels=layer_kernels,
        )
        self.emb_nodes_src = self.layer_factory.Linear(self.in_channels_src, self.hidden_dim)

    def pre_process(self, x):
        x_src, x_dst = x
        return self.emb_nodes_src(x_src)

    def post_process(self, x_dst):
        return x_dst

    def forward(
        self,
        x: PairTensor,
        batch_size: int,
        shard_info: BipartiteGraphShardInfo,
        edge_attr: Optional[Tensor] = None,
        edge_index: Optional[Adj] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        keep_x_dst_sharded: bool = False,
        **kwargs,
    ) -> PairTensor:
        x_dst = super().forward(
            x,
            batch_size,
            shard_info,
            edge_attr,
            edge_index,
            model_comm_group,
            keep_x_dst_sharded,
            **kwargs,
        )
        return x[0], x_dst


class PointWiseBackwardMapper(PointWiseMapper):
    """PointWise Mapper from hidden -> data."""

    def __init__(
        self,
        *,
        in_channels_src: int,
        in_channels_dst: int,
        hidden_dim: int,
        out_channels_dst: int,
        initialise_data_extractor_zero: bool = False,
        cpu_offload: bool = False,
        gradient_checkpointing: bool = True,
        layer_kernels: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            in_channels_src=in_channels_src,
            in_channels_dst=in_channels_dst,
            hidden_dim=hidden_dim,
            cpu_offload=cpu_offload,
            gradient_checkpointing=gradient_checkpointing,
            layer_kernels=layer_kernels,
        )
        self.out_channels_dst = out_channels_dst

        LayerNorm = self.layer_factory.LayerNorm
        Linear = self.layer_factory.Linear
        self.node_data_extractor = nn.Sequential(
            LayerNorm(normalized_shape=self.hidden_dim),
            Linear(self.hidden_dim, self.out_channels_dst),
        )
        if initialise_data_extractor_zero:
            for module in self.node_data_extractor.modules():
                if isinstance(module, nn.Linear):
                    nn.init.constant_(module.weight, 0.0)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)

    def pre_process(self, x):
        x_src, x_dst = x
        return x_src

    def post_process(self, x_dst):
        return self.node_data_extractor(x_dst)


class TransformerBaseMapper(BaseMapper, ABC):
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
        attn_channels: Optional[int] = None,
        window_size: Optional[int] = None,
        dropout_p: float = 0.0,
        qk_norm: bool = False,
        attention_implementation: str = "flash_attention",
        softcap: Optional[float] = None,
        use_alibi_slopes: bool = False,
        use_rotary_embeddings: bool = False,
        cpu_offload: bool = False,
        layer_kernels: DotDict,
        **kwargs,
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
        attn_channels : int, optional
            Internal attention width used for q/k/v projections. If None,
            defaults to the hidden dimension. This allows reducing the number
            of channels used for the attention computation without changing
            the width of the surrounding MLPs.
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
            **kwargs,
        )

        self.proc = TransformerMapperBlock(
            num_channels=hidden_dim,
            hidden_dim=mlp_hidden_ratio * hidden_dim,
            attn_channels=attn_channels,
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
        shard_info: BipartiteGraphShardInfo,
        model_comm_group: Optional[ProcessGroup] = None,
        keep_x_dst_sharded: bool = False,
        cond: Optional[tuple[Tensor, Tensor]] = None,
    ) -> PairTensor:
        x_src, x_dst = x
        shard_sizes_src, shard_sizes_dst = shard_info.src_nodes, shard_info.dst_nodes

        # Ensure src and dst are sharded
        x_src, shard_sizes_src = ensure_sharded(x_src, 0, shard_sizes_src, model_comm_group)
        x_dst, shard_sizes_dst = ensure_sharded(x_dst, 0, shard_sizes_dst, model_comm_group)

        shard_info = BipartiteGraphShardInfo(
            src_nodes=shard_sizes_src,
            dst_nodes=shard_sizes_dst,
            edges=shard_info.edges,
        )

        x_src, x_dst = self.pre_process((x_src, x_dst))

        (x_src, x_dst), _ = self.proc(
            (x_src, x_dst),
            shard_info,
            batch_size,
            model_comm_group,
            cond=cond,
        )

        x_dst = self.post_process(x_dst)

        if not keep_x_dst_sharded:
            x_dst = gather_tensor(x_dst, 0, shard_info.dst_nodes, model_comm_group)

        return x_dst

    def forward(
        self,
        x: PairTensor,
        batch_size: int,
        shard_info: BipartiteGraphShardInfo,
        edge_attr: Optional[Tensor] = None,
        edge_index: Optional[Adj] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        keep_x_dst_sharded: bool = False,
        **kwargs,
    ) -> PairTensor:
        return maybe_checkpoint(
            self.mapper_forward,
            self.gradient_checkpointing,
            x=x,
            batch_size=batch_size,
            shard_info=shard_info,
            model_comm_group=model_comm_group,
            keep_x_dst_sharded=keep_x_dst_sharded,
            **kwargs,
        )


class TransformerForwardMapper(TransformerBaseMapper):
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
        attn_channels: Optional[int] = None,
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
        attn_channels : int, optional
            Internal attention width used for q/k/v projections. If None,
            defaults to the hidden dimension. This allows reducing the number
            of channels used for the attention computation without changing
            the width of the surrounding MLPs.
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
            attn_channels=attn_channels,
            window_size=window_size,
            dropout_p=dropout_p,
            qk_norm=qk_norm,
            attention_implementation=attention_implementation,
            softcap=softcap,
            use_alibi_slopes=use_alibi_slopes,
            use_rotary_embeddings=use_rotary_embeddings,
            **kwargs,
        )

        self.emb_nodes_src = nn.Linear(self.in_channels_src, self.hidden_dim)

    def pre_process(self, x):
        x_src, x_dst = x
        x_src = self.emb_nodes_src(x_src)
        x_dst = self.emb_nodes_dst(x_dst)
        return x_src, x_dst

    def post_process(self, x_dst, **kwargs):
        return x_dst

    def forward(
        self,
        x: PairTensor,
        batch_size: int,
        shard_info: BipartiteGraphShardInfo,
        edge_attr: Optional[Tensor] = None,
        edge_index: Optional[Adj] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        keep_x_dst_sharded: bool = False,
        **kwargs,
    ) -> PairTensor:
        x_dst = super().forward(
            x,
            batch_size,
            shard_info,
            edge_attr,
            edge_index,
            model_comm_group,
            keep_x_dst_sharded,
            **kwargs,
        )
        return x[0], x_dst


class TransformerBackwardMapper(TransformerBaseMapper):
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
        attn_channels: Optional[int] = None,
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
        attn_channels : int, optional
            Internal attention width used for q/k/v projections. If None,
            defaults to the hidden dimension. This allows reducing the number
            of channels used for the attention computation without changing
            the width of the surrounding MLPs.
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
            attn_channels=attn_channels,
            window_size=window_size,
            dropout_p=dropout_p,
            qk_norm=qk_norm,
            attention_implementation=attention_implementation,
            softcap=softcap,
            use_alibi_slopes=use_alibi_slopes,
            use_rotary_embeddings=use_rotary_embeddings,
            **kwargs,
        )

        self.node_data_extractor = nn.Sequential(
            nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.out_channels_dst)
        )

    def pre_process(self, x):
        x_src, x_dst = x
        x_dst = self.emb_nodes_dst(x_dst)
        return x_src, x_dst

    def post_process(self, x_dst):
        return self.node_data_extractor(x_dst)
