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
            Input tensor with shape ``[..., input_nodes, channels]``.
        projection_matrix : torch.Tensor
            Sparse projection matrix (assumed to be on the correct device)
        num_chunks : int
            Number of chunks to project with sparse matmul. ``1``
            projects all in one matmul.

        Returns
        -------
        torch.Tensor
            Projected tensor
        """
        input_shape = x.shape
        x = x.reshape(-1, *input_shape[-2:])

        if num_chunks == 1:
            out = self._project_flattened(x, projection_matrix)
        else:
            out = torch.cat(
                [self._project_flattened(chunk, projection_matrix) for chunk in torch.chunk(x, num_chunks, dim=0)],
                dim=0,
            )

        return out.reshape(*input_shape[:-2], *out.shape[-2:])

    def _project_flattened(self, x: torch.Tensor, projection_matrix: torch.Tensor) -> torch.Tensor:
        """Project an input whose leading dimensions have already been flattened.

        Expected x shape: [flat_batch, input_nodes, channels]
        projection_matrix shape: [output_nodes, input_nodes]
        """
        batch_size = x.shape[0]
        input_nodes = x.shape[1]
        trailing_shape = x.shape[2:]

        # [B, N, ...] -> [B, N, C]
        x_flat = x.reshape(batch_size, input_nodes, -1)

        # Move batch/features into the dense RHS columns:
        # [B, N, C] -> [N, B, C] -> [N, B*C]
        x_rhs = x_flat.permute(1, 0, 2).reshape(input_nodes, -1)

        with torch.amp.autocast(device_type=x.device.type, enabled=self.autocast):
            # One sparse matmul instead of B sparse matmuls:
            # [M, N] @ [N, B*C] -> [M, B*C]
            out = torch.sparse.mm(projection_matrix, x_rhs)

        output_nodes = out.shape[0]

        # [M, B*C] -> [M, B, C] -> [B, M, C] -> [B, M, ...]
        out = out.reshape(output_nodes, batch_size, -1).permute(1, 0, 2)
        return out.reshape(batch_size, output_nodes, *trailing_shape)
