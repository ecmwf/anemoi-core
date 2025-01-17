# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import ABC
from abc import abstractmethod
from typing import Optional

from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.typing import Adj
from torch_geometric.typing import OptPairTensor
from torch_geometric.typing import Size

from anemoi.models.layers.block import GraphConvProcessorBlock
from anemoi.models.layers.block import GraphTransformerProcessorBlock
from anemoi.models.layers.block import TransformerProcessorBlock
from anemoi.models.layers.mlp import MLP

LOGGER = logging.getLogger(__name__)


class BaseProcessorChunk(nn.Module, ABC):
    """Base Processor Chunk."""

    def __init__(
        self,
        num_channels: int,
        num_layers: int,
        *args,
        activation: str = "GELU",
        **kwargs,
    ) -> None:
        """Initialize BaseProcessorChunk."""
        super().__init__()

        self.num_channels = num_channels
        self.num_layers = num_layers

    def build_blocks(self, block: nn.Module, *args, **kwargs) -> None:
        """Build Layers."""
        self.blocks = nn.ModuleList(
            [
                block(
                    *args,
                    **kwargs,
                )
                for _ in range(self.num_layers)
            ],
        )

    @abstractmethod
    def forward(
        self, x: Tensor, shapes: list, batch_size: int, model_comm_group: Optional[ProcessGroup] = None
    ) -> Tensor: ...


class TransformerProcessorChunk(BaseProcessorChunk):
    """Wraps transformer blocks for checkpointing in Processor."""

    def __init__(
        self,
        num_channels: int,
        num_layers: int,
        window_size: int,
        num_heads: int = 16,
        mlp_hidden_ratio: int = 4,
        activation: str = "GELU",
        dropout_p: float = 0.0,
        attention_implementation: str = "flash_attention",
        softcap: float = None,
        use_alibi_slopes: bool = None,
    ) -> None:
        """Initialize TransformerProcessor.

        Parameters
        ----------
        num_channels : int
            Number of channels
        num_layers : int
            Number of layers
        num_heads: int
            Number of heads to use, default 16
        mlp_hidden_ratio: int
            ratio of mlp hidden dimension to embedding dimension, default 4
        activation : str, optional
            Activation function, by default "GELU"
        dropout_p: float
            Dropout probability used for multi-head self attention, default 0.0
        attention_implementation: str, optional
            A predefined string which selects which underlying attention
            implementation, by default "flash_attention"
        softcap : float, optional
            Anything > 0 activates softcapping flash attention, by default None
        use_alibi_slopes : bool, optional
            Use aLiBI option, only used for flash attention, by default None
        """
        super().__init__(num_channels=num_channels, num_layers=num_layers)

        self.build_blocks(
            TransformerProcessorBlock,
            num_channels=num_channels,
            hidden_dim=(mlp_hidden_ratio * num_channels),
            num_heads=num_heads,
            activation=activation,
            window_size=window_size,
            dropout_p=dropout_p,
            attention_implementation=attention_implementation,
            softcap=softcap,
            use_alibi_slopes=use_alibi_slopes,
        )

    def forward(
        self, x: Tensor, shapes: list, batch_size: int, model_comm_group: Optional[ProcessGroup] = None
    ) -> Tensor:
        for i in range(self.num_layers):
            x = self.blocks[i](x, shapes, batch_size, model_comm_group=model_comm_group)

        return (x,)  # return tuple for consistency with other processors


class GNNProcessorChunk(BaseProcessorChunk):
    """Wraps edge embedding message passing blocks for checkpointing in Processor."""

    def __init__(
        self,
        num_channels: int,
        num_layers: int,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        edge_dim: Optional[int] = None,
    ) -> None:
        """Initialize GNNProcessorChunk.

        Parameters
        ----------
        num_channels : int
            Channels of the message passing blocks.
        num_layers : int
            Number of message passing blocks.
        mlp_extra_layers : int, optional
            Extra num_layers in MLP, by default 0
        activation : str, optional
            Activation function, by default "SiLU"
        edge_dim: int, by default None
            Embed edges with input dimension edge_dim,
            if None: assume embedding is not required
        """
        super().__init__(num_channels=num_channels, num_layers=num_layers)

        if edge_dim:
            self.emb_edges = MLP(
                in_features=edge_dim,
                hidden_dim=num_channels,
                out_features=num_channels,
                n_extra_layers=mlp_extra_layers,
                activation=activation,
            )
        else:
            self.emb_edges = None

        self.build_blocks(
            GraphConvProcessorBlock,
            num_channels,
            num_channels,
            mlp_extra_layers=mlp_extra_layers,
            activation=activation,
        )

    def forward(
        self,
        x: OptPairTensor,
        edge_attr: Tensor,
        edge_index: Adj,
        shapes: tuple,
        model_comm_group: Optional[ProcessGroup] = None,
        size: Optional[Size] = None,
    ) -> OptPairTensor:
        x_out = x * 1.0  # required for pytorch >= 2.1
        if self.emb_edges:
            edge_attr = self.emb_edges(edge_attr)

        for i in range(self.num_layers):
            x_out, edge_attr = self.blocks[i](x_out, edge_attr, edge_index, shapes, model_comm_group, size=size)

        return x_out, edge_attr


class GraphTransformerProcessorChunk(BaseProcessorChunk):
    """Wraps graph transformer blocks for checkpointing in Processor."""

    def __init__(
        self,
        num_channels: int,
        num_layers: int,
        num_heads: int = 16,
        mlp_hidden_ratio: int = 4,
        activation: str = "GELU",
        edge_dim: Optional[int] = None,
    ) -> None:
        """Initialize GraphTransformerProcessorChunk.

        Parameters
        ----------
        num_channels : int
            Number of channels.
        num_layers : int
            Number of layers.
        num_heads: int
            Number of heads to use, default 16
        mlp_hidden_ratio: int
            ratio of mlp hidden dimension to embedding dimension, default 4
        activation : str, optional
            Activation function, by default "GELU"
        edge_dim: int, by default None
            Embed edges with input dimension edge_dim
        """
        super().__init__(num_channels=num_channels, num_layers=num_layers)

        self.build_blocks(
            GraphTransformerProcessorBlock,
            in_channels=num_channels,
            hidden_dim=mlp_hidden_ratio * num_channels,
            out_channels=num_channels,
            num_heads=num_heads,
            edge_dim=edge_dim,
            activation=activation,
        )

    def forward(
        self,
        x: OptPairTensor,
        edge_attr: Tensor,
        edge_index: Adj,
        shapes: tuple,
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
        size: Optional[Size] = None,
    ) -> OptPairTensor:
        for i in range(self.num_layers):
            x, edge_attr = self.blocks[i](x, edge_attr, edge_index, shapes, batch_size, model_comm_group, size=size)

        return x, edge_attr
