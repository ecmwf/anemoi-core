# (C) Copyright 2024 Anemoi contributors.
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
from torch.nn.functional import dropout
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptPairTensor
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.utils import scatter
from torch_geometric.utils import softmax

from anemoi.models.layers.mlp import MLP
from anemoi.utils.config import DotDict


class GraphConv(MessagePassing):
    """Message passing module for convolutional node and edge interactions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer_kernels: DotDict,
        mlp_extra_layers: int = 0,
        **kwargs,
    ) -> None:
        """Initialize GraphConv node interactions.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
        mlp_extra_layers : int, optional
            Extra layers in MLP, by default 0
        """
        super().__init__(**kwargs)

        self.edge_mlp = MLP(
            3 * in_channels,
            out_channels,
            out_channels,
            layer_kernels=layer_kernels,
            n_extra_layers=mlp_extra_layers,
        )

    def forward(self, x: OptPairTensor, edge_attr: Tensor, edge_index: Adj, size: Optional[Size] = None):
        dim_size = x.shape[0] if isinstance(x, Tensor) else x[1].shape[0]

        out, edges_new = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size, dim_size=dim_size)

        return out, edges_new

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor, dim_size: Optional[int] = None) -> Tensor:
        edges_new = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=1)) + edge_attr

        return edges_new

    def aggregate(self, edges_new: Tensor, edge_index: Adj, dim_size: Optional[int] = None) -> tuple[Tensor, Tensor]:
        out = scatter(edges_new, edge_index[1], dim=0, dim_size=dim_size, reduce="sum")

        return out, edges_new


class GraphTransformerConv(MessagePassing):
    """Message passing part of graph transformer operator.

    Adapted from 'Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification'
    (https://arxiv.org/abs/2009.03509)
    """

    def __init__(
        self,
        out_channels: int,
        dropout: float = 0.0,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.out_channels = out_channels
        self.dropout = dropout

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        edge_attr: OptTensor,
        edge_index: Adj,
        size: Optional[Size] = None,
    ):
        dim_size = query.shape[0]
        heads = query.shape[1]

        out = self.propagate(
            edge_index=edge_index,
            size=size,
            dim_size=dim_size,
            edge_attr=edge_attr,
            heads=heads,
            query=query,
            key=key,
            value=value,
        )

        return out

    def message(
        self,
        heads: int,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        if edge_attr is not None:
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / self.out_channels**0.5

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = dropout(alpha, p=self.dropout, training=self.training)

        return (value_j + edge_attr) * alpha.view(-1, heads, 1)
