from __future__ import annotations

import torch

from anemoi.models.layers.graph_provider import ProjectionGraphProvider


class SparseProjector(torch.nn.Module):
    """Applies sparse projection matrix to input tensors.

    Stateless projector that receives matrix as input to forward().
    """

    def __init__(self, autocast: bool = False) -> None:
        """Initialize SparseProjector.

        Parameters
        ----------
        autocast : bool
            Use automatic mixed precision
        """
        super().__init__()
        self.autocast = autocast

    def forward(self, x: torch.Tensor, projection_matrix: torch.Tensor) -> torch.Tensor:
        """Apply sparse projection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        projection_matrix : torch.Tensor
            Sparse projection matrix (assumed to be on correct device)

        Returns
        -------
        torch.Tensor
            Projected tensor
        """
        out = []
        with torch.amp.autocast(device_type=x.device.type, enabled=self.autocast):
            for i in range(x.shape[0]):
                out.append(torch.sparse.mm(projection_matrix, x[i, ...]))
        return torch.stack(out)


def _projection_matrix(
    projection: ProjectionGraphProvider | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(projection, ProjectionGraphProvider):
        return projection.get_edges(device=device)
    return projection.to(device=device)


def apply_sparse_projector_with_reshaping(
    projector: SparseProjector,
    x: torch.Tensor,
    projection: ProjectionGraphProvider | torch.Tensor,
) -> torch.Tensor:
    """Project trailing ``[grid, variables]`` dimensions."""
    input_shape = x.shape
    x = x.reshape(-1, *input_shape[-2:])
    x = projector(x, _projection_matrix(projection, x.device))
    return x.reshape(*input_shape[:-2], *x.shape[-2:])
