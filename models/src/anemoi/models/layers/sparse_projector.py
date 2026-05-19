from __future__ import annotations

import torch


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

    def apply_with_provider(self, batch: torch.Tensor, provider: object) -> torch.Tensor:
        """Apply projection via *provider*, handling arbitrary leading dimensions.

        Parameters
        ----------
        batch:
            Input tensor of shape ``[..., nodes, vars]``.
        provider:
            Object with a ``get_edges(device=...)`` method returning the sparse matrix.

        Returns
        -------
        torch.Tensor
            Projected tensor of shape ``[..., dst_nodes, vars]``.
        """
        return apply_sparse_projector_with_reshaping(self, batch, provider)


def _projection_matrix(
    projection: object | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(projection, torch.Tensor):
        return projection.to(device=device)
    return projection.get_edges(device=device)


def apply_sparse_projector_with_reshaping(
    projector: SparseProjector,
    x: torch.Tensor,
    projection: object | torch.Tensor,
) -> torch.Tensor:
    """Project trailing ``[grid, variables]`` dimensions."""
    input_shape = x.shape
    x = x.reshape(-1, *input_shape[-2:])
    x = projector(x, _projection_matrix(projection, x.device))
    return x.reshape(*input_shape[:-2], *x.shape[-2:])
