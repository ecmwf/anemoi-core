from typing import Optional

import einops
import torch
from torch import nn

from anemoi.models.distributed.graph import gather_channels
from anemoi.models.distributed.graph import shard_channels
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.layers.sparse_projector import SparseProjector


class SkipConnection(nn.Module):
    """Skip connection module that selects the most recent timestep from the input sequence.

    This module is used to bypass processing layers and directly pass the latest input forward.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        x = x[:, -1, ...]  # pick current date
        return x


class NoConnection(nn.Module):
    """No-op connection that returns a zero tensor with the same shape as the last timestep."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        x = x[:, -1, ...]  # pick current date
        return torch.zeros_like(x, device=x.device, dtype=x.dtype)


class TruncatedConnection(nn.Module):
    """Applies a coarse-graining and reconstruction of input features using sparse projections to
    truncate high frequency features.

    This module uses two projection operators: one to map features from the full-resolution
    grid to a truncated (coarse) grid, and another to project back to the original resolution.

    Parameters
    ----------
        num_data_nodes (int): Number of nodes in the original full-resolution grid.
        num_truncation_nodes (int): Number of nodes in the truncated grid.
        sub_graph_down: Graph object containing edge_index and edge_length for down-projection.
        sub_graph_up: Graph object containing edge_index and edge_length for up-projection.
        edge_weight_attribute (str, optional): Name of the edge attribute to use as weights for the projections.
    """

    def __init__(
        self,
        graph,
        data_nodes: str,
        truncation_nodes: str,
        edge_weight_attribute: Optional[str] = None,
        src_node_weight_attribute: Optional[str] = None,
        autocast: bool = False,
    ) -> None:
        super().__init__()

        num_data_nodes = graph[data_nodes].num_nodes
        num_truncation_nodes = graph[truncation_nodes].num_nodes
        sub_graph_down = graph[data_nodes, "to", truncation_nodes]
        sub_graph_up = graph[truncation_nodes, "to", data_nodes]

        up_weight = torch.ones(sub_graph_up.edge_index.shape[1], device=sub_graph_up.edge_index.device)
        down_weight = torch.ones(sub_graph_down.edge_index.shape[1], device=sub_graph_down.edge_index.device)

        if edge_weight_attribute:
            up_weight = sub_graph_up[edge_weight_attribute].squeeze() * up_weight
            down_weight = sub_graph_down[edge_weight_attribute].squeeze() * down_weight

        if src_node_weight_attribute:
            down_weight = graph[data_nodes][src_node_weight_attribute][sub_graph_down.edge_index[0]]
            up_weight = graph[truncation_nodes][src_node_weight_attribute][sub_graph_up.edge_index[0]]

        self.project_up = SparseProjector(
            edge_index=sub_graph_up.edge_index,
            weights=up_weight,
            src_size=num_truncation_nodes,
            dst_size=num_data_nodes,
            autocast=autocast,
        )

        self.project_down = SparseProjector(
            edge_index=sub_graph_down.edge_index,
            weights=down_weight,
            src_size=num_data_nodes,
            dst_size=num_truncation_nodes,
            autocast=autocast,
        )

    def forward(self, x, grid_shard_shapes=None, model_comm_group=None, *args, **kwargs):
        batch_size = x.shape[0]
        x = x[:, -1, ...]  # pick latest step
        shard_shapes = apply_shard_shapes(x, 0, grid_shard_shapes) if grid_shard_shapes is not None else None

        x = einops.rearrange(x, "batch ensemble grid features -> (batch ensemble) grid features")
        x = self._to_channel_shards(x, shard_shapes, model_comm_group)
        x = self.project_down(x)
        x = self.project_up(x)
        x = self._to_grid_shards(x, shard_shapes, model_comm_group)
        x = einops.rearrange(x, "(batch ensemble) grid features -> batch ensemble grid features", batch=batch_size)

        return x

    def _to_channel_shards(self, x, shard_shapes=None, model_comm_group=None):
        return self._reshard(x, shard_channels, shard_shapes, model_comm_group)

    def _to_grid_shards(self, x, shard_shapes=None, model_comm_group=None):
        return self._reshard(x, gather_channels, shard_shapes, model_comm_group)

    def _reshard(self, x, fn, shard_shapes=None, model_comm_group=None):
        if shard_shapes is not None:
            x = fn(x, shard_shapes, model_comm_group)
        return x
