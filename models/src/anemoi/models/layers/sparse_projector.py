# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from anemoi.models.layers.graph_provider import ProjectionGraphProvider


class SparseProjector(torch.nn.Module):
    """Applies the projection of a :class:`ProjectionGraphProvider`.

    The provider's backend picks the implementation:

    - ``"torch"``: CSR matrix and ``torch.sparse.mm``, on any device.
    - ``"triton"``: padded table and the kernels in :mod:`anemoi.models.triton.sparse_projection`.
      Uses less memory; the backward pass uses atomic adds, so it is not bitwise reproducible.

    The provider is passed to :meth:`forward`; the projector holds no state.
    """

    def __init__(self, autocast: bool = False, num_chunks: int = 1) -> None:
        """Initialize SparseProjector.

        Parameters
        ----------
        autocast : bool
            Use autocast for the torch backend's sparse matmul.
        num_chunks : int
            Default number of chunks for the leading dimensions.
        """
        super().__init__()
        self.autocast = autocast
        self.num_chunks = num_chunks

    def forward(
        self,
        x: torch.Tensor,
        provider: ProjectionGraphProvider,
        num_chunks: int | None = None,
        matrix_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Apply the projection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``[..., input_nodes, channels]``.
        provider : ProjectionGraphProvider
            The projection, with ``input_nodes`` columns.
        num_chunks : int, optional
            Number of chunks to project the leading dimensions in. ``1``
            projects all at once. Defaults to the value configured
            when constructing the projector.
        matrix_dtype : torch.dtype, optional
            Torch backend only: dtype to cast the matrix to. Triton always uses float32 weights.

        Returns
        -------
        torch.Tensor
            Projected tensor
        """
        num_chunks = self.num_chunks if num_chunks is None else num_chunks
        num_targets, num_sources = provider.shape
        if x.shape[-2] != num_sources:
            raise ValueError(
                f"The projection reads {num_sources} source nodes, but the input has {x.shape[-2]} "
                f"(shape {tuple(x.shape)})."
            )

        if provider.backend == "triton":
            if not x.is_cuda:
                raise ValueError(
                    "The triton backend of the sparse projection needs the input on a GPU; build the model with "
                    "the 'torch' backend, or switch a built model with "
                    "anemoi.models.layers.graph_provider.use_torch_projections."
                )
            from anemoi.models.triton.sparse_projection import project_triton

            neighbours, weights, row_lengths = provider.get_table(device=x.device)

            def project(flat: torch.Tensor) -> torch.Tensor:
                return project_triton(flat, neighbours, weights, row_lengths)

        else:
            matrix = provider.get_edges(device=x.device, dtype=matrix_dtype)

            def project(flat: torch.Tensor) -> torch.Tensor:
                return self._project_flattened(flat, matrix)

        input_shape = x.shape
        # Explicit sizes so that a field with zero channels reshapes too.
        x = x.reshape(math.prod(input_shape[:-2]), *input_shape[-2:])

        try:
            if num_chunks == 1:
                out = project(x)
            else:
                out = torch.cat([project(chunk) for chunk in torch.chunk(x, num_chunks, dim=0)], dim=0)
        except torch.cuda.OutOfMemoryError as e:
            raise torch.cuda.OutOfMemoryError(
                f"Out of memory during sparse projection. Consider doubling the number of chunks (currently {num_chunks}). This will reduce memory usage at the cost of additional kernel launches, which can increase runtime. Original error: {e}"
            ) from e

        return out.reshape(*input_shape[:-2], num_targets, input_shape[-1])

    def _project_flattened(self, x: torch.Tensor, projection_matrix: torch.Tensor) -> torch.Tensor:
        """Torch backend: project ``[flat_batch, input_nodes, channels]`` with the CSR ``projection_matrix``."""
        batch_size, input_nodes, channels = x.shape

        # Move batch/features into the dense RHS columns:
        # [B, N, C] -> [N, B, C] -> [N, B*C]
        x_rhs = x.permute(1, 0, 2).reshape(input_nodes, batch_size * channels)

        with torch.amp.autocast(device_type=x.device.type, enabled=self.autocast):
            # One sparse matmul instead of B sparse matmuls:
            # [M, N] @ [N, B*C] -> [M, B*C]
            out = torch.sparse.mm(projection_matrix, x_rhs)

        output_nodes = out.shape[0]

        # [M, B*C] -> [M, B, C] -> [B, M, C]
        return out.reshape(output_nodes, batch_size, channels).permute(1, 0, 2)
