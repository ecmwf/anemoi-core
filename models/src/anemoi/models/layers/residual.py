import einops
import torch
from torch import nn

from anemoi.models.distributed.graph import gather_channels
from anemoi.models.distributed.graph import shard_channels
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.distributed.shapes import get_shard_shapes


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


class TruncationMapper(nn.Module):
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
    """

    def __init__(
        self,
        num_data_nodes,
        num_truncation_nodes,
        sub_graph_down,
        sub_graph_up,
    ) -> None:
        super().__init__()

        self.project_down = SparseProjector(
            edge_index=sub_graph_down.edge_index,
            weights=sub_graph_down.edge_length.squeeze(),
            src_size=num_data_nodes,
            dst_size=num_truncation_nodes,
        )
        self.project_up = SparseProjector(
            edge_index=sub_graph_up.edge_index,
            weights=sub_graph_up.edge_length.squeeze(),
            src_size=num_truncation_nodes,
            dst_size=num_data_nodes,
        )

    def forward(self, x, grid_shard_shapes=None, model_comm_group=None, *args, **kwargs):
        self.to(x.device)
        batch_size = x.shape[0]
        x = x[:, -1, ...]  # pick current date

        x = einops.rearrange(x, "batch ensemble grid features -> (batch ensemble) grid features")
        x = self._to_channel_shards(x, grid_shard_shapes, model_comm_group)
        x = self.project_down(x)
        x = self.project_up(x)
        x = self._to_grid_shards(x, grid_shard_shapes, model_comm_group)
        x = einops.rearrange(x, "(batch ensemble) grid features -> batch ensemble grid features", batch=batch_size)

        return x

    def _to_channel_shards(self, x, grid_shard_shapes=None, model_comm_group=None):
        return self._reshard(x, shard_channels, grid_shard_shapes, model_comm_group)

    def _to_grid_shards(self, x, grid_shard_shapes=None, model_comm_group=None):
        return self._reshard(x, gather_channels, grid_shard_shapes, model_comm_group)

    def _reshard(self, x, fn, grid_shard_shapes=None, model_comm_group=None):
        if grid_shard_shapes is not None:
            shard_shapes = self._get_shard_shapes(x, 0, grid_shard_shapes, model_comm_group)
            x = fn(x, shard_shapes, model_comm_group)
        return x

    def _get_shard_shapes(self, x, dim=0, shard_shapes_dim=None, model_comm_group=None):
        if shard_shapes_dim is None:
            return get_shard_shapes(x, dim, model_comm_group)
        else:
            return apply_shard_shapes(x, dim, shard_shapes_dim)


class SparseProjector(nn.Module):
    """Constructs and applies a sparse projection matrix for mapping features between grids.

    The projection matrix is constructed from edge indices and edge attributes (e.g., distances),
    with optional Gaussian weighting and row normalization.

    Parameters
    ----------
        edge_index (Tensor): Edge indices (2, E) representing source and destination nodes.
        weights (Tensor): Raw edge attributes (e.g., distances) of shape (E,).
        src_size (int): Number of nodes in the source grid.
        dst_size (int): Number of nodes in the target grid.
        apply_gaussian_weighting (bool): Whether to apply Gaussian weighting to the raw edge values.
        row_normalize (bool): Whether to normalize weights per destination node.
    """

    def __init__(self, edge_index, weights, src_size, dst_size, apply_gaussian_weighting=True, row_normalize=True):
        super().__init__()
        weights = _gaussian_distance_weighting(weights) if apply_gaussian_weighting else weights
        weights = _row_normalize_weights(edge_index, weights, dst_size) if row_normalize else weights

        self.projection_matrix = torch.sparse_coo_tensor(
            edge_index,
            weights,
            (src_size, dst_size),
            device=edge_index.device,
        ).coalesce()

    def forward(self, x, *args, **kwargs):
        out = []
        for i in range(x.shape[0]):
            out.append(torch.sparse.mm(self.projection_matrix.T, x[i, ...]))
        return torch.stack(out)


def _row_normalize_weights(
    edge_index,
    weights,
    num_target_nodes,
):
    total = torch.zeros(num_target_nodes, device=weights.device)
    norm = total.scatter_add_(0, edge_index[1].long(), weights)
    norm = norm[edge_index[1]]
    return weights / (norm + 1e-8)


def _gaussian_distance_weighting(weights: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Apply Gaussian distance weighting to the weights."""
    return torch.exp(-0.5 * (weights / sigma) ** 2)
