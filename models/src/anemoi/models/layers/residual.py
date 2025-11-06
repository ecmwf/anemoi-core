from abc import abstractmethod
from typing import Optional

import einops
import torch
from torch import nn

from anemoi.models.distributed.graph import gather_channels
from anemoi.models.distributed.graph import shard_channels
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.layers.sparse_projector import build_sparse_projector


class BaseResidualConnection(nn.Module):
    """Base class for residual connection modules."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def get_last_timestep(self, x):
        # x shape: (batch, time, ens, nodes, features)
        return x[:, -1, ...]  # pick current date

    @abstractmethod
    def forward(self, x, *args, **kwargs):
        pass


class SkipConnection(BaseResidualConnection):
    """Skip connection module

    It selects the most recent timestep from the input sequence.

    This module is used to bypass processing layers and directly pass the latest input forward.
    """

    def forward(self, x, *args, **kwargs):
        return self.get_last_timestep(x)


class NoConnection(BaseResidualConnection):
    """No-op connection

    This module returns a zero tensor with the same shape as the last timestep.
    """

    def forward(self, x, *args, **kwargs):
        x = self.get_last_timestep(x)
        return torch.zeros_like(x, device=x.device, dtype=x.dtype)


class TruncatedConnection(nn.Module):
    """Truncated skip connection

    It applies a coarse-graining and reconstruction of input features using sparse projections to
    truncate high frequency features.

    This module uses two projection operators: one to map features from the full-resolution
    grid to a truncated (coarse) grid, and another to project back to the original resolution.

    Parameters
    ----------
    graph : HeteroData
        The graph containing the subgraphs for down and up projections.
    data_nodes : str
        Name of the nodes representing the data nodes.
    truncation_nodes : str
        Name of the nodes representing the truncated (coarse) nodes.
    edge_weight_attribute : str, optional
        Name of the edge attribute to use as weights for the projections.
    src_node_weight_attribute : str, optional
        Name of the source node attribute to use as weights for the projections.
    autocast : bool, default False
        Whether to use automatic mixed precision for the projections.
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

        self.project_down = build_sparse_projector(
            graph=graph,
            edges_name=(data_nodes, "to", truncation_nodes),
            edge_weight_attribute=edge_weight_attribute,
            src_node_weight_attribute=src_node_weight_attribute,
            autocast=autocast,
        )

        self.project_up = build_sparse_projector(
            graph=graph,
            edges_name=(truncation_nodes, "to", data_nodes),
            edge_weight_attribute=edge_weight_attribute,
            src_node_weight_attribute=src_node_weight_attribute,
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
