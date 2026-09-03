# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Optional

from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.typing import Adj

from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.processor import GraphTransformerProcessor
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class GraphTransformerPreMixer(nn.Module):
    """Nonlinear point-to-point mixing on the data grid, ahead of the encoder.

    Motivation
    ----------
    The encoder mapper is a *single* attention block, so each hidden node is

        out_h = sum_{s in N(h)} alpha_hs * V(x_s),   V(x_s) = Linear(LayerNorm(Linear(x_s)))

    Every nonlinearity is per-token and the aggregation is a softmax-weighted
    sum, so a hidden node can only ever hold a data-dependent weighted *mean*
    of its neighbourhood. Sub-grid variance, gradients and texture are not
    representable, because they need products of *distinct* points' features.

    This module runs graph-attention layers over a data -> data graph before
    that pooling, so each point token becomes a nonlinear function of its own
    neighbourhood. The encoder then pools nonlinear local descriptors rather
    than linear point features, which is a strictly richer class of set
    function (mean-of-nonlinear-local-features, not nonlinear-of-mean).

    Shape and initialisation contract
    ---------------------------------
    The mixed signal is added back residually at the *input* width, so
    ``input_dim`` is unchanged and every downstream module (encoder, decoder)
    keeps its shapes. With ``initialise_out_zero`` the module is an exact
    identity at initialisation, so a checkpoint trained without a pre-mixer can
    be forked and reproduces its parent bit-for-bit on step 0.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_channels: int,
        num_layers: int,
        num_chunks: int,
        num_heads: int,
        mlp_hidden_ratio: float,
        edge_dim: int,
        attn_channels: Optional[int] = None,
        qk_norm: bool = False,
        initialise_out_zero: bool = True,
        cpu_offload: bool = False,
        gradient_checkpointing: bool = True,
        graph_attention_backend: str = "triton",
        edge_pre_mlp: bool = False,
        layer_kernels: DotDict,
        **kwargs,  # accept unused extras like sub_graph_edge_attributes / trainable_size
    ) -> None:
        """Initialize GraphTransformerPreMixer.

        Parameters
        ----------
        in_channels : int
            Width of the data-node features, i.e. the model's ``input_dim``.
            The module reads and writes at this width.
        num_channels : int
            Internal mixing width. Independent of ``model.num_channels``: the
            data grid is several times larger than the hidden grid, so this is
            normally set well below it.
        num_layers : int
            Number of graph-attention layers of point-to-point mixing.
        num_chunks : int
            Number of gradient-checkpointing chunks. Set equal to ``num_layers``
            to checkpoint every layer.
        num_heads : int
            Number of attention heads.
        mlp_hidden_ratio : float
            Ratio of mlp hidden dimension to embedding dimension.
        edge_dim : int
            Edge feature dimension of the data -> data graph.
        attn_channels : int, optional
            Internal attention width for q/k/v, by default None (= num_channels).
        qk_norm : bool, optional
            Normalize query and key, by default False.
        initialise_out_zero : bool, optional
            Zero-initialise the output projection so the module starts as an
            exact identity, by default True. Keep this True when forking a
            checkpoint trained without a pre-mixer.
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False.
        gradient_checkpointing : bool, optional
            Whether to enable gradient checkpointing, by default True.
        graph_attention_backend : str, optional
            Backend for the graph attention, "triton" or "pyg", by default "triton".
        edge_pre_mlp : bool, optional
            Allow for edge feature mixing, by default False.
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_channels = num_channels
        self.layer_factory = load_layer_kernels(layer_kernels)

        self.emb_in = self.layer_factory.Linear(in_channels, num_channels)

        self.proc = GraphTransformerProcessor(
            num_layers=num_layers,
            num_channels=num_channels,
            num_chunks=num_chunks,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            edge_dim=edge_dim,
            attn_channels=attn_channels,
            qk_norm=qk_norm,
            cpu_offload=cpu_offload,
            gradient_checkpointing=gradient_checkpointing,
            layer_kernels=layer_kernels,
            graph_attention_backend=graph_attention_backend,
            edge_pre_mlp=edge_pre_mlp,
        )

        self.emb_out = self.layer_factory.Linear(num_channels, in_channels)
        if initialise_out_zero:
            nn.init.zeros_(self.emb_out.weight)
            if self.emb_out.bias is not None:
                nn.init.zeros_(self.emb_out.bias)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_info: GraphShardInfo,
        edge_attr: Tensor,
        edge_index: Adj,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> Tensor:
        """Mix data-node features with those of their neighbours.

        Parameters
        ----------
        x : Tensor
            Data node features of width ``in_channels``.
        batch_size : int
            Batch size.
        shard_info : GraphShardInfo
            Shard metadata for the data nodes and the data -> data edges.
            ``nodes=None`` means the data grid is replicated, not sharded.
        edge_attr : Tensor
            Edge attributes of the data -> data graph.
        edge_index : Adj
            Edge indices of the data -> data graph.
        model_comm_group : ProcessGroup, optional
            Model communication group.

        Returns
        -------
        Tensor
            Mixed features, same shape and shard layout as ``x``.
        """
        x_mixed = self.proc(
            self.emb_in(x),
            batch_size=batch_size,
            shard_info=shard_info,
            edge_attr=edge_attr,
            edge_index=edge_index,
            model_comm_group=model_comm_group,
        )

        return x + self.emb_out(x_mixed)
