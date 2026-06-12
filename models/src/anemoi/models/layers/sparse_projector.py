import torch


class SparseProjector(torch.nn.Module):
    """Applies a sparse projection matrix to input tensors.

    Stateless: the matrix is passed to :meth:`forward`, not stored.
    """

    def __init__(self, autocast: bool = False) -> None:
        super().__init__()
        self.autocast = autocast

    def forward(self, x: torch.Tensor, projection_matrix: torch.Tensor) -> torch.Tensor:
        """Project ``x`` through ``projection_matrix`` (assumed to be on the right device)."""
        out = []
        with torch.amp.autocast(device_type=x.device.type, enabled=self.autocast):
            for i in range(x.shape[0]):
                out.append(torch.sparse.mm(projection_matrix, x[i, ...]))
        return torch.stack(out)

    def project(self, batch: torch.Tensor, provider: object) -> torch.Tensor:
        """Project ``batch`` of shape ``[..., nodes, vars]`` through the *provider*'s matrix.

        Handles arbitrary leading dimensions; the *provider* supplies the matrix via ``get_edges(device=...)``.
        """
        input_shape = batch.shape
        batch = batch.reshape(-1, *input_shape[-2:])
        projection_matrix = provider.get_edges(device=batch.device)
        batch = self(batch, projection_matrix)
        return batch.reshape(*input_shape[:-2], *batch.shape[-2:])
