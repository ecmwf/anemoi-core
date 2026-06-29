# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch


class SparseProjector(torch.nn.Module):
    """Applies a sparse projection matrix to input tensors.

    Stateless: the matrix is passed to :meth:`forward`, not stored.
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

    def forward(
        self,
        x: torch.Tensor,
        projection_matrix: torch.Tensor,
        num_chunks: int = 1,
    ) -> torch.Tensor:
        """Apply sparse projection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        projection_matrix : torch.Tensor
            Sparse projection matrix (assumed to be on the correct device)
        num_chunks : int
            Number of batch chunks to project with sparse matmul. ``1``
            projects the full batch in one matmul.

        Returns
        -------
        torch.Tensor
            Projected tensor
        """
        if num_chunks == 1:
            return self._forward(x, projection_matrix)

        return torch.cat(
            [self._forward(chunk, projection_matrix) for chunk in torch.chunk(x, num_chunks, dim=0)],
            dim=0,
        )

    def _forward(self, x: torch.Tensor, projection_matrix: torch.Tensor) -> torch.Tensor:
        """Apply sparse projection.

        Expected x shape: [batch, input_nodes, ...]
        projection_matrix shape: [output_nodes, input_nodes]
        """
        batch_size = x.shape[0]
        input_nodes = x.shape[1]
        trailing_shape = x.shape[2:]

        with torch.amp.autocast(device_type=x.device.type, enabled=self.autocast):
            # [B, N, ...] -> [B, N, C]
            x_flat = x.reshape(batch_size, input_nodes, -1)

            # Move batch/features into the dense RHS columns:
            # [B, N, C] -> [N, B, C] -> [N, B*C]
            x_rhs = x_flat.permute(1, 0, 2).reshape(input_nodes, -1)

            # One sparse matmul instead of B sparse matmuls:
            # [M, N] @ [N, B*C] -> [M, B*C]
            out = torch.sparse.mm(projection_matrix, x_rhs)

            output_nodes = out.shape[0]

            # [M, B*C] -> [M, B, C] -> [B, M, C] -> [B, M, ...]
            out = out.reshape(output_nodes, batch_size, -1).permute(1, 0, 2)
            out = out.reshape(batch_size, output_nodes, *trailing_shape)

        return out

    def project(self, batch: torch.Tensor, provider: object, num_chunks: int = 1) -> torch.Tensor:
        """Project ``batch`` of shape ``[..., nodes, vars]`` through the *provider*'s matrix.

        Handles arbitrary leading dimensions; the *provider* supplies the matrix via ``get_edges(device=...)``.
        """
        input_shape = batch.shape
        batch = batch.reshape(-1, *input_shape[-2:])
        projection_matrix = provider.get_edges(device=batch.device)
        batch = self(batch, projection_matrix, num_chunks=num_chunks)
        return batch.reshape(*input_shape[:-2], *batch.shape[-2:])
