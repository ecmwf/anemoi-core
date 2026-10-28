# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import scatter

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.graph import sync_tensor
from anemoi.models.layers.layout import GraphLayout
from anemoi.utils.config import DotDict


class EdgeDepthwiseConv(nn.Module):
    """A learned stencil per channel over the edges of a graph.

    The counterpart of ``GeocyclicDepthwiseConv`` on an unstructured mesh. A linear layer turns
    each edge's attributes into one weight per channel, and each node's new value is the mean over
    its incoming edges of the neighbour's value times that weight:

        w_ijc = Linear(edge_attr_ij)[c],    out_ic = (1 / deg_i) * sum_j w_ijc * x_jc

    Like the square stencil on a grid, this is linear in the state, acts on every channel on its
    own, and reaches only the neighbours of a node. It is directional, because the weights are
    built from the edge attributes, which carry the length and direction of each edge.
    """

    def __init__(self, num_channels: int, edge_dim: int, layer_kernels: DotDict) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.weight = layer_kernels.Linear(edge_dim, num_channels, bias=False)

    def forward(self, x: Tensor, layout: GraphLayout) -> Tensor:
        """Apply the stencil to x, of shape (batch * local nodes, channels)."""
        # Each GPU holds the edges of its own destination nodes, so it needs every node's value.
        source, destination = layout.edge_index
        values = sync_tensor(x, 0, layout.shard_info.nodes, layout.model_comm_group)
        messages = self.weight(layout.edge_attr) * values[source]
        out = scatter(messages, destination, dim=0, dim_size=layout.size, reduce="mean")
        # sync_tensor already all-reduces the gradient, so this must not gather it again.
        return shard_tensor(out, 0, layout.shard_info.nodes, layout.model_comm_group, gather_in_backward=False)
