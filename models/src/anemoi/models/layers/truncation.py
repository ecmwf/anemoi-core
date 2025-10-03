# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np
import torch

from anemoi.models.distributed.graph import gather_channels
from anemoi.models.distributed.graph import shard_channels
from anemoi.models.distributed.shapes import get_or_apply_shard_shapes

LOGGER = logging.getLogger(__name__)


class BaseTruncation:
    """Apply resolution truncation/upsampling via sparse projection matrices.

    This utility holds two (optional) sparse COO matrices:

    - ``A_down``: projects from a high-resolution representation to a coarse one
      (e.g., spectral/graph truncation).
    - ``A_up``: projects from the coarse representation back to high resolution
      (e.g., zero-padding or learned up-projection).

    Both matrices are expected in SciPy CSR/COO-like format at construction time
    and are converted to PyTorch sparse tensors. During ``__call__`` the
    matrices are moved to the input device (first use) and applied per sample in
    the batch. When inputs are grid-sharded across ranks, tensors are reshaped
    to channel-sharding to apply the projection on the full sequence and then
    restored to their original sharding scheme.

    Notes
    -----
    - Sparse tensors are **not** registered as buffers because DDP does not
      reliably broadcast sparse tensors; instead the matrices are lazily moved
      to the correct device on first use.
    - Matrix–tensor multiplication is performed as ``A @ X`` (left
      multiplication), where ``A`` is sparse (``[n_out, n_in]``) and ``X`` is
      dense (``[n_in, d]``), producing ``[n_out, d]``.
    """

    def __init__(self, truncation_data: dict) -> None:
        """Build the truncation matrices.

        Parameters
        ----------
        truncation_data : dict
            Dictionary possibly containing keys ``"down"`` and/or ``"up"`` with
            SciPy sparse matrices. ``"down"`` defines the high→coarse projection
            (stored as ``A_down``); ``"up"`` defines the coarse→high projection
            (stored as ``A_up``).
        """
        self.A_down, self.A_up = None, None
        if "down" in truncation_data:
            self.A_down = self._make_truncation_matrix(truncation_data["down"])
            LOGGER.info("Truncation: A_down %s", self.A_down.shape)
        if "up" in truncation_data:
            self.A_up = self._make_truncation_matrix(truncation_data["up"])
            LOGGER.info("Truncation: A_up %s", self.A_up.shape)

    def _make_truncation_matrix(self, A, data_type=torch.float32):
        """Convert a SciPy sparse matrix to a coalesced PyTorch COO tensor.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
            Input sparse matrix with shape ``(n_out, n_in)``.
        data_type : torch.dtype, optional
            Target dtype for the tensor values, by default ``torch.float32``.

        Returns
        -------
        torch.Tensor
            A coalesced sparse COO tensor with the same shape as ``A``.
        """
        A_ = torch.sparse_coo_tensor(
            torch.tensor(np.vstack(A.nonzero()), dtype=torch.long),
            torch.tensor(A.data, dtype=data_type),
            size=A.shape,
        ).coalesce()
        return A_

    def _multiply_sparse(self, x, A):
        """Left-multiply a dense matrix by a sparse projection.

        Parameters
        ----------
        x : torch.Tensor
            Dense 2-D tensor with shape ``(n_in, d)``.
        A : torch.Tensor
            Sparse COO tensor with shape ``(n_out, n_in)``.

        Returns
        -------
        torch.Tensor
            Dense 2-D tensor with shape ``(n_out, d)`` equal to ``A @ x``.
        """
        return torch.sparse.mm(A, x)

    def _truncate_fields(self, x, A, batch_size=None, auto_cast=False):
        """Apply a sparse projection to each item in a batch.

        Parameters
        ----------
        x : torch.Tensor
            Dense 3-D tensor with shape ``(B, n_in, d)``. For each batch item
            ``i``, ``x[i]`` is multiplied as ``A @ x[i]``.
        A : torch.Tensor
            Sparse COO tensor with shape ``(n_out, n_in)``.
        batch_size : int, optional
            Number of batch elements to process. If ``None`` (default), uses
            ``x.shape[0]``.
        auto_cast : bool, optional
            If ``True``, enables CUDA autocast for the multiplication loop.

        Returns
        -------
        torch.Tensor
            Dense 3-D tensor with shape ``(B, n_out, d)`` containing the
            projected batch.
        """
        if not batch_size:
            batch_size = x.shape[0]
        out = []
        with torch.amp.autocast(device_type="cuda", enabled=auto_cast):
            for i in range(batch_size):
                out.append(self._multiply_sparse(x[i, ...], A))
        return torch.stack(out)

    def __call__(self, x, grid_shard_shapes=None, model_comm_group=None):
        """Apply down/up truncation to a (possibly sharded) batch.

        This function optionally:
        1) Reshapes grid-sharded inputs to channel-sharded layout to expose the
           full sequence to the projection matrices.
        2) Applies ``A_down`` (high→coarse) and/or ``A_up`` (coarse→high) per
           batch element when provided.
        3) Restores the original sharding layout.

        Parameters
        ----------
        x : torch.Tensor
            Input dense tensor of shape ``(B, n_in, d)`` if unsharded. When
            grid-sharded, the leading dimensions depend on the sharding layout;
            this method will handle reshaping internally.
        grid_shard_shapes : Any, optional
            Distributed shape metadata used to convert between grid and
            channel sharding. If ``None``, no resharding is performed.
        model_comm_group : Any, optional
            Communication group handle used by distributed helpers.

        Returns
        -------
        torch.Tensor
            Output tensor with the same global shape semantics as ``x``. If
            truncation matrices are present, the ``n_in`` dimension is replaced
            by the corresponding ``n_out`` after projection.
        """
        if self.A_down is not None or self.A_up is not None:
            if grid_shard_shapes is not None:
                shard_shapes = get_or_apply_shard_shapes(x, 0, grid_shard_shapes, model_comm_group)
                # grid-sharded input: reshard to channel-shards to apply truncation
                x = shard_channels(x, shard_shapes, model_comm_group)  # we get the full sequence here

            # these can't be registered as buffers because ddp does not like to broadcast sparse tensors
            # hence we check that they are on the correct device ; copy should only happen in the first forward run
            if self.A_down is not None:
                self.A_down = self.A_down.to(x.device)
                x = self._truncate_fields(x, self.A_down)  # to coarse resolution
            if self.A_up is not None:
                self.A_up = self.A_up.to(x.device)
                x = self._truncate_fields(x, self.A_up)  # back to high resolution

            if grid_shard_shapes is not None:
                # back to grid-sharding as before
                x = gather_channels(x, shard_shapes, model_comm_group)

        return x
