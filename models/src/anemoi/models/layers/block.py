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
from typing import Union

import einops
import torch
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.typing import Adj
from torch_geometric.typing import OptPairTensor
from torch_geometric.typing import Size

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.graph import sync_tensor
from anemoi.models.distributed.khop_edges import sort_edges_1hop_chunks
from anemoi.models.distributed.transformer import shard_heads
from anemoi.models.distributed.transformer import shard_sequence
from anemoi.models.layers.attention import MultiHeadCrossAttention
from anemoi.models.layers.attention import MultiHeadSelfAttention
from anemoi.models.layers.conv import GraphConv
from anemoi.models.layers.conv import GraphTransformerConv
from anemoi.models.layers.mlp import MLP
from anemoi.models.triton.utils import edge_index_to_csc
from anemoi.models.triton.utils import is_triton_available
from anemoi.utils.config import DotDict

if is_triton_available():
    from anemoi.models.triton.gt import GraphTransformerFunction

LOGGER = logging.getLogger(__name__)

# Number of chunks used in inference (https://github.com/ecmwf/anemoi-core/pull/66)
NUM_CHUNKS_INFERENCE = int(os.environ.get("ANEMOI_INFERENCE_NUM_CHUNKS", "1"))
NUM_CHUNKS_INFERENCE_PROCESSOR = int(os.environ.get("ANEMOI_INFERENCE_NUM_CHUNKS_PROCESSOR", NUM_CHUNKS_INFERENCE))


class BaseBlock(nn.Module, ABC):
    """Base class for network blocks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def forward(
        self,
        x: OptPairTensor,
        edge_attr: torch.Tensor,
        edge_index: Adj,
        shapes: tuple,
        batch_size: int,
        size: Optional[Size] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        **layer_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class PointWiseMLPProcessorBlock(BaseBlock):
    """Point-wise block with MLPs."""

    def __init__(self, *, num_channels: int, hidden_dim: int, layer_kernels: DotDict, dropout_p: float = 0.0):
        super().__init__()
        assert dropout_p is None or (0.0 <= dropout_p <= 1.0), "dropout_p must be in [0.0, 1.0]"
        layers = [
            layer_kernels.Linear(num_channels, hidden_dim),
            # This pattern has been proven to produce good results in point-wise models
            layer_kernels.LayerNorm(hidden_dim),
            layer_kernels.Activation(),
        ]
        if num_channels != hidden_dim:
            layers.append(layer_kernels.Linear(hidden_dim, num_channels))

        if dropout_p is not None and dropout_p > 0:
            layers.append(nn.Dropout(p=dropout_p))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x: Tensor,
        shapes: list,
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
        **layer_kwargs,
    ) -> tuple[Tensor]:
        return (self.mlp(x),)


class TransformerProcessorBlock(BaseBlock):
    """Transformer block with MultiHeadSelfAttention and MLPs."""

    def __init__(
        self,
        *,
        num_channels: int,
        hidden_dim: int,
        num_heads: int,
        window_size: int,
        layer_kernels: DotDict,
        dropout_p: float = 0.0,
        qk_norm: bool = False,
        attention_implementation: str = "flash_attention",
        softcap: Optional[float] = None,
        use_alibi_slopes: bool = False,
        use_rotary_embeddings: bool = False,
    ):
        super().__init__()

        self.layer_norm_attention = layer_kernels.LayerNorm(normalized_shape=num_channels)
        self.layer_norm_mlp = layer_kernels.LayerNorm(normalized_shape=num_channels)

        self.attention = MultiHeadSelfAttention(
            num_heads=num_heads,
            embed_dim=num_channels,
            window_size=window_size,
            qkv_bias=False,
            is_causal=False,
            qk_norm=qk_norm,
            dropout_p=dropout_p,
            layer_kernels=layer_kernels,
            attention_implementation=attention_implementation,
            softcap=softcap,
            use_alibi_slopes=use_alibi_slopes,
            use_rotary_embeddings=use_rotary_embeddings,
        )

        self.mlp = nn.Sequential(
            layer_kernels.Linear(num_channels, hidden_dim),
            layer_kernels.Activation(),
            layer_kernels.Linear(hidden_dim, num_channels),
        )

    def forward(
        self,
        x: Tensor,
        shapes: list,
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
        cond: Optional[Tensor] = None,
        **layer_kwargs,
    ) -> tuple[Tensor]:

        # In case we have conditionings we pass these to the layer norm
        cond_kwargs = {"cond": cond} if cond is not None else {}

        x = x + self.attention(
            self.layer_norm_attention(x, **cond_kwargs), shapes, batch_size, model_comm_group=model_comm_group
        )
        x = x + self.mlp(
            self.layer_norm_mlp(
                x,
                **cond_kwargs,
            )
        )
        return (x,)


class TransformerMapperBlock(TransformerProcessorBlock):
    """Transformer mapper block with MultiHeadCrossAttention and MLPs."""

    def __init__(
        self,
        *,
        num_channels: int,
        hidden_dim: int,
        num_heads: int,
        window_size: int,
        layer_kernels: DotDict,
        dropout_p: float = 0.0,
        qk_norm: bool = False,
        attention_implementation: str = "flash_attention",
        softcap: Optional[float] = None,
        use_alibi_slopes: bool = False,
        use_rotary_embeddings: bool = False,
    ):
        super().__init__(
            num_channels=num_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            window_size=window_size,
            layer_kernels=layer_kernels,
            dropout_p=dropout_p,
            qk_norm=qk_norm,
            attention_implementation=attention_implementation,
            softcap=softcap,
            use_alibi_slopes=use_alibi_slopes,
            use_rotary_embeddings=use_rotary_embeddings,
        )

        self.attention = MultiHeadCrossAttention(
            num_heads=num_heads,
            embed_dim=num_channels,
            window_size=window_size,
            qkv_bias=False,
            qk_norm=qk_norm,
            is_causal=False,
            dropout_p=dropout_p,
            layer_kernels=layer_kernels,
            attention_implementation=attention_implementation,
            softcap=softcap,
            use_alibi_slopes=use_alibi_slopes,
            use_rotary_embeddings=use_rotary_embeddings,
        )

        LayerNorm = layer_kernels.LayerNorm

        self.layer_norm_attention_src = LayerNorm(num_channels)
        self.layer_norm_attention_dst = LayerNorm(num_channels)
        self.layer_norm_mpl = LayerNorm(num_channels)

    def forward(
        self,
        x: OptPairTensor,
        shapes: list,
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[Tensor, Tensor]:
        x_src = self.layer_norm_attention_src(x[0])
        x_dst = self.layer_norm_attention_dst(x[1])
        x_dst = x_dst + self.attention((x_src, x_dst), shapes, batch_size, model_comm_group=model_comm_group)
        x_dst = x_dst + self.mlp(self.layer_norm_mpl(x_dst))
        return (x_src, x_dst), None  # logic expects return of edge_attr


class GraphConvBaseBlock(BaseBlock):
    """Message passing block with MLPs for node embeddings."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_chunks: int,
        mlp_extra_layers: int = 0,
        update_src_nodes: bool = True,
        layer_kernels: DotDict,
        edge_dim: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Initialize GNNBlock.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        num_chunks : int
            do message passing in X chunks
        mlp_extra_layers : int
            Extra layers in MLP, by default 0
        update_src_nodes: bool
            Update src if src and dst nodes are given, by default True
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
        """
        super().__init__(**kwargs)

        if edge_dim:
            self.emb_edges = MLP(
                in_features=edge_dim,
                hidden_dim=out_channels,
                out_features=out_channels,
                layer_kernels=layer_kernels,
                n_extra_layers=mlp_extra_layers,
            )
        else:
            self.emb_edges = None

        self.update_src_nodes = update_src_nodes
        self.num_chunks = num_chunks

        self.node_mlp = MLP(
            in_features=2 * in_channels,
            hidden_dim=out_channels,
            out_features=out_channels,
            layer_kernels=layer_kernels,
            n_extra_layers=mlp_extra_layers,
        )

        self.conv = GraphConv(
            in_channels=in_channels,
            out_channels=out_channels,
            layer_kernels=layer_kernels,
            mlp_extra_layers=mlp_extra_layers,
        )

    @abstractmethod
    def forward(
        self,
        x: OptPairTensor,
        edge_attr: Tensor,
        edge_index: Adj,
        shapes: tuple,
        model_comm_group: Optional[ProcessGroup] = None,
        size: Optional[Size] = None,
        **layer_kwargs,
    ) -> tuple[Tensor, Tensor]: ...


class GraphConvProcessorBlock(GraphConvBaseBlock):
    def forward(
        self,
        x: OptPairTensor,
        edge_attr: Tensor,
        edge_index: Adj,
        shapes: tuple,
        model_comm_group: Optional[ProcessGroup] = None,
        size: Optional[Size] = None,
        **layer_kwargs,
    ) -> tuple[Tensor, Tensor]:
        if self.emb_edges is not None:
            edge_attr = self.emb_edges(edge_attr)

        x_in = sync_tensor(x, 0, shapes[1], model_comm_group)

        if self.num_chunks > 1:
            edge_index_list = torch.tensor_split(edge_index, self.num_chunks, dim=1)
            edge_attr_list = torch.tensor_split(edge_attr, self.num_chunks, dim=0)
            edges_out = []
            for i in range(self.num_chunks):
                out1, edges_out1 = self.conv(x_in, edge_attr_list[i], edge_index_list[i], size=size)
                edges_out.append(edges_out1)
                if i == 0:
                    out = torch.zeros_like(out1)
                out = out + out1
            edges_new = torch.cat(edges_out, dim=0)
        else:
            out, edges_new = self.conv(x_in, edge_attr, edge_index, size=size)

        out = shard_tensor(out, 0, shapes[1], model_comm_group, gather_in_backward=False)

        nodes_new = self.node_mlp(torch.cat([x, out], dim=1)) + x

        return nodes_new, edges_new


class GraphConvMapperBlock(GraphConvBaseBlock):

    def __ini__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_chunks: int,
        mlp_extra_layers: int = 0,
        update_src_nodes: bool = True,
        layer_kernels: DotDict,
        **kwargs,
    ) -> None:
        """Initialize GNN Mapper Block.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        num_chunks : int
            Number of chunks
        mlp_extra_layers : int, optional
            Extra layers in MLP, by default 0
        update_src_nodes : bool, optional
            Update src if src and dst nodes are given, by default True
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
        kwargs : dict
            Additional arguments for the base class.
        """
        super().__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            layer_kernels=layer_kernels,
            mlp_extra_layers=mlp_extra_layers,
            update_src_nodes=update_src_nodes,
            num_chunks=num_chunks,
            **kwargs,
        )

    def forward(
        self,
        x: OptPairTensor,
        edge_attr: Tensor,
        edge_index: Adj,
        shapes: tuple,
        model_comm_group: Optional[ProcessGroup] = None,
        size: Optional[Size] = None,
        **layer_kwargs,
    ) -> tuple[Tensor, Tensor]:

        x_src = sync_tensor(x[0], 0, shapes[0], model_comm_group)
        x_dst = sync_tensor(x[1], 0, shapes[1], model_comm_group)
        x_in = (x_src, x_dst)

        if self.num_chunks > 1:
            edge_index_list = torch.tensor_split(edge_index, self.num_chunks, dim=1)
            edge_attr_list = torch.tensor_split(edge_attr, self.num_chunks, dim=0)
            edges_out = []
            for i in range(self.num_chunks):
                out1, edges_out1 = self.conv(x_in, edge_attr_list[i], edge_index_list[i], size=size)
                edges_out.append(edges_out1)
                if i == 0:
                    out = torch.zeros_like(out1)
                out = out + out1
            edges_new = torch.cat(edges_out, dim=0)
        else:
            out, edges_new = self.conv(x_in, edge_attr, edge_index, size=size)

        out = shard_tensor(out, 0, shapes[1], model_comm_group, gather_in_backward=False)

        nodes_new_dst = self.node_mlp(torch.cat([x[1], out], dim=1)) + x[1]

        # update only needed in forward mapper
        nodes_new_src = x[0] if not self.update_src_nodes else self.node_mlp(torch.cat([x[0], x[0]], dim=1)) + x[0]

        nodes_new = (nodes_new_src, nodes_new_dst)

        return nodes_new, edges_new


class GraphTransformerBaseBlock(BaseBlock, ABC):
    """Message passing block with MLPs for node embeddings."""

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        num_heads: int,
        edge_dim: int,
        bias: bool = True,
        qk_norm: bool = False,
        update_src_nodes: bool = False,
        layer_kernels: DotDict,
        graph_attention_backend: str = "triton",
        edge_pre_mlp: bool = False,
        **kwargs,
    ) -> None:
        """Initialize GraphTransformerBlock.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        num_heads : int,
            Number of heads
        edge_dim : int,
            Edge dimension
        bias : bool, by default True,
            Add bias or not
        qk_norm : bool, by default False
            Normalize query and key
        update_src_nodes: bool, by default False
            Update src if src and dst nodes are given
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
        graph_attention_backend: str, by default "triton"
            Backend to use for graph transformer conv, options are "triton" and "pyg"
        edge_pre_mlp: bool, by default False
            Allow for edge feature mixing
        """
        super().__init__(**kwargs)

        self.update_src_nodes = update_src_nodes

        self.out_channels_conv = out_channels // num_heads
        self.num_heads = num_heads
        self.qk_norm = qk_norm

        Linear = layer_kernels.Linear
        LayerNorm = layer_kernels.LayerNorm
        Activation = layer_kernels.Activation
        self.lin_key = Linear(in_channels, num_heads * self.out_channels_conv)
        self.lin_query = Linear(in_channels, num_heads * self.out_channels_conv)
        self.lin_value = Linear(in_channels, num_heads * self.out_channels_conv)
        self.lin_self = Linear(in_channels, num_heads * self.out_channels_conv, bias=bias)
        self.lin_edge = Linear(edge_dim, num_heads * self.out_channels_conv)  # , bias=False)

        self.projection = Linear(out_channels, out_channels)

        if self.qk_norm:
            self.q_norm = layer_kernels.QueryNorm(self.out_channels_conv)
            self.k_norm = layer_kernels.KeyNorm(self.out_channels_conv)

        self.layer_norm_attention = LayerNorm(normalized_shape=in_channels)
        self.layer_norm_mlp_dst = LayerNorm(normalized_shape=out_channels)
        self.node_dst_mlp = nn.Sequential(
            Linear(out_channels, hidden_dim),
            Activation(),
            Linear(hidden_dim, out_channels),
        )

        # Optional edge preprocessing MLP
        if edge_pre_mlp:
            self.edge_pre_mlp = nn.Sequential(
                Linear(edge_dim, edge_dim),
                Activation(),
            )
        else:
            self.edge_pre_mlp = nn.Identity()

        self.graph_attention_backend = graph_attention_backend
        assert self.graph_attention_backend in [
            "triton",
            "pyg",
        ], f"Backend {self.graph_attention_backend} not supported for GraphTransformerBlock, valid options are 'triton' and 'pyg'"

        if not is_triton_available():
            LOGGER.warning(
                f"{self.__class__.__name__} requested the triton graph attention backend but triton is not available. Falling back to 'pyg' backend."
            )
            self.graph_attention_backend = "pyg"

        if self.graph_attention_backend == "triton":
            LOGGER.info(f"{self.__class__.__name__} using triton graph attention backend.")
            self.conv = GraphTransformerFunction.apply
        else:
            self.conv = GraphTransformerConv(out_channels=self.out_channels_conv)

    def run_node_dst_mlp(self, x, **layer_kwargs):
        return self.node_dst_mlp(self.layer_norm_mlp_dst(x, **layer_kwargs))

    def get_qkve(
        self,
        x: OptPairTensor,
        edge_attr: Tensor,
    ):
        x_src, x_dst = x if isinstance(x, tuple) else (x, x)

        query = self.lin_query(x_dst)
        key = self.lin_key(x_src)
        value = self.lin_value(x_src)
        edges = self.lin_edge(self.edge_pre_mlp(edge_attr))

        return query, key, value, edges

    def shard_qkve_heads(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        edges: Tensor,
        shapes: tuple,
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Shards qkv and edges along head dimension."""
        if model_comm_group is not None:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), "Only batch size of 1 is supported when model is sharded across GPUs"

        shape_src_nodes, shape_dst_nodes, shape_edges = shapes

        query, key, value, edges = (
            einops.rearrange(
                t,
                "(batch grid) (heads vars) -> batch heads grid vars",
                heads=self.num_heads,
                vars=self.out_channels_conv,
                batch=batch_size,
            )
            for t in (query, key, value, edges)
        )
        query = shard_heads(query, shapes=shape_dst_nodes, mgroup=model_comm_group)
        key = shard_heads(key, shapes=shape_src_nodes, mgroup=model_comm_group)
        value = shard_heads(value, shapes=shape_src_nodes, mgroup=model_comm_group)
        edges = shard_heads(edges, shapes=shape_edges, mgroup=model_comm_group)

        query, key, value, edges = (
            einops.rearrange(t, "batch heads grid vars -> (batch grid) heads vars") for t in (query, key, value, edges)
        )

        return query, key, value, edges

    def apply_gt(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        edges: Tensor,
        edge_index: Adj,
        size: Union[int, tuple[int, int]],
    ) -> Tensor:
        # self.conv requires size to be a tuple
        conv_size = (size, size) if isinstance(size, int) else size

        if self.graph_attention_backend == "triton":
            csc, perm, reverse = edge_index_to_csc(edge_index, num_nodes=conv_size, reverse=True)
            edges_csc = edges.index_select(0, perm)
            args_conv = (edges_csc, csc, reverse)
        else:
            args_conv = (edges, edge_index, conv_size)

        return self.conv(query, key, value, *args_conv)

    def attention_block(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        edges: Tensor,
        edge_index: Adj,
        size: Union[int, tuple[int, int]],
        num_chunks: int,
    ) -> Tensor:
        # split 1-hop edges into chunks, compute self.conv chunk-wise
        if num_chunks > 1:
            edge_attr_list, edge_index_list = sort_edges_1hop_chunks(
                num_nodes=size, edge_attr=edges, edge_index=edge_index, num_chunks=num_chunks
            )
            # shape: (num_nodes, num_heads, out_channels_conv)
            out = torch.zeros((*query.shape[:-1], self.out_channels_conv), device=query.device)
            for i in range(num_chunks):
                out += self.apply_gt(
                    query=query, key=key, value=value, edges=edge_attr_list[i], edge_index=edge_index_list[i], size=size
                )
        else:
            out = self.apply_gt(query=query, key=key, value=value, edges=edges, edge_index=edge_index, size=size)

        return out

    def shard_output_seq(
        self,
        out: Tensor,
        shapes: tuple,
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> Tensor:
        """Shards Tensor sequence dimension."""
        shape_dst_nodes = shapes[1]

        out = einops.rearrange(out, "(batch grid) heads vars -> batch heads grid vars", batch=batch_size)
        out = shard_sequence(out, shapes=shape_dst_nodes, mgroup=model_comm_group)
        out = einops.rearrange(out, "batch heads grid vars -> (batch grid) (heads vars)")

        return out

    @abstractmethod
    def forward(
        self,
        x: OptPairTensor,
        edge_attr: Tensor,
        edge_index: Adj,
        shapes: tuple,
        batch_size: int,
        size: Union[int, tuple[int, int]],
        model_comm_group: Optional[ProcessGroup] = None,
        **kwargs,
    ): ...


class GraphTransformerMapperBlock(GraphTransformerBaseBlock):
    """Graph Transformer Block for node embeddings."""

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        num_heads: int,
        edge_dim: int,
        bias: bool = True,
        qk_norm: bool = False,
        update_src_nodes: bool = False,
        layer_kernels: DotDict,
        shard_strategy: str = "edges",
        graph_attention_backend: str = "triton",
        edge_pre_mlp: bool = False,
        **kwargs,
    ) -> None:
        """Initialize GraphTransformerBlock.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        hidden_dim : int
            Hidden dimension
        out_channels : int
            Number of output channels.
        num_heads : int,
            Number of heads
        edge_dim : int,
            Edge dimension
        bias : bool
            Apply bias in layers, by default Tru
        qk_norm: bool
            Normalize query and key, by default False
        update_src_nodes: bool
            Update src if src and dst nodes are given, by default False
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
        shard_strategy: str, by default "edges"
            Strategy to shard tensors
        graph_attention_backend: str, by default "triton"
            Backend to use for graph transformer conv, options are "triton" and "pyg"
        edge_pre_mlp: bool, by default False
            Allow for edge feature mixing
        """

        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            edge_dim=edge_dim,
            layer_kernels=layer_kernels,
            num_heads=num_heads,
            bias=bias,
            qk_norm=qk_norm,
            update_src_nodes=update_src_nodes,
            graph_attention_backend=graph_attention_backend,
            edge_pre_mlp=edge_pre_mlp,
            **kwargs,
        )

        Linear = layer_kernels.Linear
        LayerNorm = layer_kernels.LayerNorm

        self.layer_norm_attention_src = LayerNorm(normalized_shape=in_channels)
        self.layer_norm_attention_dest = self.layer_norm_attention

        if self.update_src_nodes:
            self.layer_norm_mlp_src = LayerNorm(normalized_shape=out_channels)
            self.node_src_mlp = nn.Sequential(
                Linear(out_channels, hidden_dim),
                layer_kernels.Activation(),
                Linear(hidden_dim, out_channels),
            )
        else:
            self.layer_norm_mlp_src = nn.Identity()
            self.node_src_mlp = nn.Identity()

        self.shard_strategy = shard_strategy

    def run_node_src_mlp(self, x, **layer_kwargs):
        return self.node_src_mlp(self.layer_norm_mlp_src(x, **layer_kwargs))

    def prepare_qkve_edge_sharding(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        edges: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            einops.rearrange(
                t,
                "nodes (heads vars) -> nodes heads vars",
                heads=self.num_heads,
                vars=self.out_channels_conv,
            )
            for t in (query, key, value, edges)
        )

    def forward(
        self,
        x: OptPairTensor,
        edge_attr: Tensor,
        edge_index: Adj,
        shapes: tuple,
        batch_size: int,
        size: Union[int, tuple[int, int]],
        model_comm_group: Optional[ProcessGroup] = None,
        cond: Optional[tuple[Tensor, Tensor]] = None,
        **layer_kwargs,
    ):
        x_skip = x

        # In case we have conditionings we pass these to the layer norm
        cond_src_kwargs = {"cond": cond[0]} if cond is not None else {}
        cond_dst_kwargs = {"cond": cond[1]} if cond is not None else {}

        x = (
            self.layer_norm_attention_src(x[0], **cond_src_kwargs),
            self.layer_norm_attention_dest(x[1], **cond_dst_kwargs),
        )

        x_r = self.lin_self(x[1])

        query, key, value, edges = self.get_qkve(x, edge_attr)

        if self.shard_strategy == "heads":
            query, key, value, edges = self.shard_qkve_heads(
                query, key, value, edges, shapes, batch_size, model_comm_group
            )
        else:
            query, key, value, edges = self.prepare_qkve_edge_sharding(query, key, value, edges)

        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)

        out = self.attention_block(query, key, value, edges, edge_index, size, num_chunks=1)

        if self.shard_strategy == "heads":
            out = self.shard_output_seq(out, shapes, batch_size, model_comm_group)
        else:
            out = einops.rearrange(out, "nodes heads vars -> nodes (heads vars)")

        out = self.projection(out + x_r)
        out = out + x_skip[1]

        nodes_new_dst = self.run_node_dst_mlp(out, **cond_dst_kwargs) + out

        if self.update_src_nodes:
            nodes_new_src = self.run_node_src_mlp(x_skip[0], **cond_dst_kwargs) + x_skip[0]
        else:
            nodes_new_src = x_skip[0]

        nodes_new = (nodes_new_src, nodes_new_dst)

        return nodes_new, edge_attr


class GraphTransformerProcessorBlock(GraphTransformerBaseBlock):
    """Graph Transformer Block for node embeddings."""

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        num_heads: int,
        edge_dim: int,
        bias: bool = True,
        qk_norm: bool = False,
        update_src_nodes: bool = False,
        layer_kernels: DotDict,
        graph_attention_backend: str = "triton",
        edge_pre_mlp: bool = False,
        **kwargs,
    ) -> None:
        """Initialize GraphTransformerBlock.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        num_heads : int,
            Number of heads
        edge_dim : int,
            Edge dimension
        bias : bool
            Add bias or not, by default True
        qk_norm: bool
            Normalize query and key, by default False
        update_src_nodes: bool
            Update src if src and dst nodes are given, by default False
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
        graph_attention_backend: str, by default "triton"
            Backend to use for graph transformer conv, options are "triton" and "pyg"
        edge_pre_mlp: bool, by default False
            Allow for edge feature mixing
        """

        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            edge_dim=edge_dim,
            layer_kernels=layer_kernels,
            num_heads=num_heads,
            bias=bias,
            qk_norm=qk_norm,
            update_src_nodes=update_src_nodes,
            graph_attention_backend=graph_attention_backend,
            edge_pre_mlp=edge_pre_mlp,
            **kwargs,
        )

    def forward(
        self,
        x: OptPairTensor,
        edge_attr: Tensor,
        edge_index: Adj,
        shapes: tuple,
        batch_size: int,
        size: Union[int, tuple[int, int]],
        model_comm_group: Optional[ProcessGroup] = None,
        cond: Optional[Tensor] = None,
    ):
        x_skip = x

        # In case we have conditionings we pass these to the layer norm
        cond_kwargs = {"cond": cond} if cond is not None else {}

        x = self.layer_norm_attention(x, **cond_kwargs)
        x_r = self.lin_self(x)

        query, key, value, edges = self.get_qkve(x, edge_attr)

        query, key, value, edges = self.shard_qkve_heads(query, key, value, edges, shapes, batch_size, model_comm_group)

        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)

        # "inner" chunking for memory reductions in inference, controlled via env variable:
        num_chunks = 1 if self.training else NUM_CHUNKS_INFERENCE_PROCESSOR

        out = self.attention_block(query, key, value, edges, edge_index, size, num_chunks)

        out = self.shard_output_seq(out, shapes, batch_size, model_comm_group)

        # out = self.projection(out + x_r) in chunks:
        out = torch.cat([self.projection(chunk) for chunk in torch.tensor_split(out + x_r, num_chunks, dim=0)], dim=0)

        out = out + x_skip
        nodes_new = self.run_node_dst_mlp(out, **cond_kwargs) + out

        return nodes_new, edge_attr
