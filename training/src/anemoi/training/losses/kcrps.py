# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Literal

import einops
import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.training.losses.base import BaseLoss

LOGGER = logging.getLogger(__name__)


CRPSBackend = Literal["naive", "stable"]


class CRPS(BaseLoss):
    """Kernel CRPS loss for ensemble predictions."""

    def __init__(
        self,
        alpha: float = 0.95,
        backend: CRPSBackend = "stable",
        no_autocast: bool = True,
        ignore_nans: bool = False,
        **kwargs,  # noqa: ARG002
    ) -> None:
        """Latitude- and (inverse-)variance-weighted kernel CRPS loss.

        ``alpha`` controls the interpolation between standard CRPS and fair CRPS. ``alpha=0`` gives standard
        CRPS, ``alpha=1`` gives fair CRPS, and values between 0 and 1 give the almost fair CRPS formulation.

        Parameters
        ----------
        alpha : float
            Factor for linear combination of fair (unbiased, ensemble variance component weighted by (ens-size-1)^-1)
            and standard CRPS (1.0 = fully fair, 0.0 = fully unfair)
        backend : {"naive", "stable"}
            Backend used for the point-wise CRPS calculation. The naive backend uses a simple loop over unordered
            ensemble-member pairs and avoids materializing the full pairwise tensor. The stable backend materializes
            pairwise tensors and uses the numerically stable all-pairs formulation.
        no_autocast : bool, optional
            Deactivate autocast for the kernel CRPS calculation
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False
        """
        super().__init__(ignore_nans=ignore_nans)

        self._validate_arguments(alpha, backend)

        self.alpha = alpha
        self.backend = backend
        self.no_autocast = no_autocast

    @staticmethod
    def _validate_arguments(alpha: float, backend: CRPSBackend) -> None:
        """Validate CRPS constructor arguments."""
        if not 0.0 <= alpha <= 1.0:
            msg = f"alpha must be in the range [0, 1], got {alpha}"
            raise ValueError(msg)
        if backend not in ("naive", "stable"):
            msg = f"Unknown CRPS backend {backend!r}. Expected one of: 'naive', 'stable'."
            raise ValueError(msg)

    def _kernel_crps(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        alpha: float | None = None,
        backend: CRPSBackend | None = None,
    ) -> torch.Tensor:
        """Kernel (ensemble) CRPS.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted ensemble, shape (batch_size, n_out_steps, n_vars, latlon, ens_size)
        targets : torch.Tensor
            Ground truth, shape (batch_size, n_out_steps, n_vars, latlon)
        alpha : float
            Factor for linear combination of fair (unbiased, ensemble variance component weighted by (ens-size-1)^-1)
            and standard CRPS (1.0 = fully fair, 0.0 = fully unfair)
        backend : {"naive", "stable"}
            Backend used for the point-wise CRPS calculation.

        Returns
        -------
        CRPS : torch.Tensor
            The point-wise kernel CRPS, shape (batch_size, n_out_steps, n_vars, latlon).
        """
        alpha = self.alpha if alpha is None else alpha
        backend = self.backend if backend is None else backend
        ens_size = preds.shape[-1]
        assert ens_size > 1, "Ensemble size must be greater than 1."

        if backend == "naive":
            return self._kernel_crps_naive(preds, targets, alpha)
        if backend == "stable":
            return self._kernel_crps_stable(preds, targets, alpha)

        msg = f"Unknown CRPS backend {backend!r}. Expected one of: 'naive', 'stable'."
        raise ValueError(msg)

    @staticmethod
    def _kernel_crps_naive(preds: torch.Tensor, targets: torch.Tensor, alpha: float) -> torch.Tensor:
        """CRPS formulation using a simple loop over unordered ensemble-member pairs."""
        ens_size = preds.shape[-1]

        mae = torch.mean(torch.abs(targets[..., None] - preds), dim=-1)
        coef = -(alpha / (ens_size * (ens_size - 1)) + (1.0 - alpha) / (ens_size**2))

        ens_var = torch.zeros_like(mae)
        for i in range(ens_size - 1):
            ens_var += torch.sum(torch.abs(preds[..., i].unsqueeze(-1) - preds[..., i + 1 :]), dim=-1)

        return mae + coef * ens_var

    @staticmethod
    def _kernel_crps_stable(preds: torch.Tensor, targets: torch.Tensor, alpha: float) -> torch.Tensor:
        """CRPS formulation materializing pairwise tensors for the all-pairs formulation."""
        ens_size = preds.shape[-1]

        epsilon = (1.0 - alpha) / ens_size

        var = torch.abs(preds.unsqueeze(dim=-1) - preds.unsqueeze(dim=-2))
        diag = torch.eye(ens_size, dtype=torch.bool, device=preds.device)
        err_r = einops.repeat(
            torch.abs(preds - targets.unsqueeze(dim=-1)),
            "batch t var latlon ens -> batch t var latlon n ens",
            n=ens_size,
        )

        mem_err = err_r * ~diag
        mem_err_transpose = mem_err.transpose(-1, -2)

        coef = 1.0 / (2.0 * ens_size * (ens_size - 1))
        return coef * torch.sum(mem_err + mem_err_transpose - (1 - epsilon) * var, dim=(-1, -2))

    def forward(
        self,
        y_pred: torch.Tensor,
        y_target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        squash_mode: str = "sum",
    ) -> torch.Tensor:
        is_sharded = grid_shard_slice is not None

        y_target = einops.rearrange(y_target, "bs t latlon v -> bs t v latlon")
        y_pred = einops.rearrange(y_pred, "bs t e latlon v -> bs t v latlon e")

        if self.no_autocast:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                crps = self._kernel_crps(y_pred, y_target)
        else:
            crps = self._kernel_crps(y_pred, y_target)

        crps = einops.rearrange(crps, "bs t v latlon -> bs t 1 latlon v")
        crps = self.scale(crps, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)

        return self.reduce(crps, squash=squash, squash_mode=squash_mode, group=group if is_sharded else None)

    @property
    def name(self) -> str:
        return f"crps{self.alpha:.2f}"
