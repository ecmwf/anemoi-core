# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from contextlib import nullcontext
from typing import Literal

import einops
import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import Squash_mode
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


CRPSBackend = Literal["naive", "stable"]


def antithetic_cross_pairs(ens_size: int, device: torch.device | None = None) -> torch.Tensor:
    """The ordered member pairs (i, j), i != j, that are not antithetic partners.

    With noise centring, members (0, 1), (2, 3), ... are mirror images of each other (the noise injectors pair
    global members ``2k`` and ``2k + 1``), so the distance between partners is not that of two independent members:
    for a linear response it is sqrt(2) times larger. A last, unpaired member (odd ``ens_size``) pairs with all
    others.

    Parameters
    ----------
    ens_size : int
        Number of ensemble members, in their global order.
    device : torch.device, optional
        Device of the returned mask.

    Returns
    -------
    torch.Tensor
        Boolean mask of shape (ens_size, ens_size), True for the pairs to use in the spread term.
    """
    index = torch.arange(ens_size, device=device)
    cross = index[:, None] // 2 != index[None, :] // 2
    if not cross.any():
        msg = f"An ensemble of {ens_size} antithetic members has no independent pair for the spread term."
        raise ValueError(msg)
    return cross


class CRPS(BaseLoss):
    """Kernel CRPS loss for ensemble predictions."""

    # Also for subclasses that set up the CRPS attributes without calling this __init__ (e.g. SpectralCRPSLoss).
    antithetic_pairs: bool = False

    def __init__(
        self,
        alpha: float = 0.95,
        backend: CRPSBackend = "stable",
        no_autocast: bool = True,
        ignore_nans: bool = False,
        antithetic_pairs: bool = False,
        **kwargs,  # noqa: ARG002
    ) -> None:
        """Latitude- and (inverse-)variance-weighted kernel CRPS loss.

        ``alpha`` controls the interpolation between standard CRPS and fair CRPS. ``alpha=0`` gives standard
        CRPS, ``alpha=1`` gives fair CRPS, and values between 0 and 1 give the almost fair CRPS formulation.

        ``antithetic_pairs`` is for ensembles whose noise is centred (mirrored member pairs): the spread term then
        averages the member distances over the pairs that are not partners (:func:`antithetic_cross_pairs`), so it
        stays an estimate of the spread of independent members. The ``alpha`` blend is unchanged: the spread term
        is ``(1 - (1 - alpha) / ens_size) / 2`` times the mean distance, as without pairs.

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
        antithetic_pairs : bool, optional
            Members (0, 1), (2, 3), ... are antithetic partners: leave those pairs out of the spread term, by
            default False
        """
        super().__init__(ignore_nans=ignore_nans)

        self._validate_arguments(alpha, backend)

        self.alpha = alpha
        self.backend = backend
        self.no_autocast = no_autocast
        self.antithetic_pairs = bool(antithetic_pairs)

    @staticmethod
    def _validate_arguments(alpha: float, backend: CRPSBackend) -> None:
        """Validate CRPS constructor arguments."""
        if not 0.0 <= alpha <= 1.0:
            msg = f"alpha must be in the range [0, 1], got {alpha}"
            raise ValueError(msg)
        if backend not in ("naive", "stable"):
            msg = f"Unknown CRPS backend {backend!r}. Expected one of: 'naive', 'stable'."
            raise ValueError(msg)

    def mask_nans(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask CRPS inputs while preserving each tensor's ensemble size."""
        target_nan_mask = torch.isnan(target).any(dim=TensorDim.ENSEMBLE_DIM, keepdim=True)
        pred_nan_mask = torch.isnan(pred).any(dim=TensorDim.ENSEMBLE_DIM, keepdim=True)
        nan_mask = target_nan_mask | pred_nan_mask
        target = target.masked_fill(nan_mask, 0.0)
        pred = pred.masked_fill(nan_mask, 0.0)
        return pred, target

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
            Predicted ensemble, shape (batch_size, n_out_steps, n_vars, latlon, ens_size).
        targets : torch.Tensor
            Ground truth, shape (batch_size, n_out_steps, n_vars, latlon, 1).
        alpha : float
            Factor for linear combination of fair (unbiased, ensemble variance component weighted by (ens-size-1)^-1)
            and standard CRPS (1.0 = fully fair, 0.0 = fully unfair).
        backend : {"naive", "stable"}
            Backend used for the point-wise CRPS calculation.

        Returns
        -------
        CRPS : torch.Tensor
            The point-wise kernel CRPS, shape (batch_size, n_out_steps, n_vars, latlon).
        """
        # Remove ensemble dim for ground truth.
        targets = targets.squeeze(-1)
        alpha = self.alpha if alpha is None else alpha
        backend = self.backend if backend is None else backend
        ens_size = preds.shape[-1]
        assert ens_size > 1, "Ensemble size must be greater than 1."

        if backend == "naive":
            return self._kernel_crps_naive(preds, targets, alpha, antithetic_pairs=self.antithetic_pairs)
        if backend == "stable":
            return self._kernel_crps_stable(preds, targets, alpha, antithetic_pairs=self.antithetic_pairs)

        msg = f"Unknown CRPS backend {backend!r}. Expected one of: 'naive', 'stable'."
        raise ValueError(msg)

    @staticmethod
    def _kernel_crps_naive(
        preds: torch.Tensor,
        targets: torch.Tensor,
        alpha: float,
        antithetic_pairs: bool = False,
    ) -> torch.Tensor:
        """CRPS formulation using a simple loop over unordered ensemble-member pairs."""
        ens_size = preds.shape[-1]

        mae = torch.mean(torch.abs(targets[..., None] - preds), dim=-1)
        if antithetic_pairs:
            pairs = torch.triu(antithetic_cross_pairs(ens_size), diagonal=1).nonzero().tolist()
            spread = torch.zeros_like(mae)
            for i, j in pairs:
                spread += torch.abs(preds[..., i] - preds[..., j])
            return mae - 0.5 * (1.0 - (1.0 - alpha) / ens_size) * spread / len(pairs)

        coef = -(alpha / (ens_size * (ens_size - 1)) + (1.0 - alpha) / (ens_size**2))

        ens_var = torch.zeros_like(mae)
        for i in range(ens_size - 1):
            ens_var += torch.sum(torch.abs(preds[..., i].unsqueeze(-1) - preds[..., i + 1 :]), dim=-1)

        return mae + coef * ens_var

    @staticmethod
    def _kernel_crps_stable(
        preds: torch.Tensor,
        targets: torch.Tensor,
        alpha: float,
        antithetic_pairs: bool = False,
    ) -> torch.Tensor:
        """CRPS formulation materializing pairwise tensors for the all-pairs formulation."""
        ens_size = preds.shape[-1]

        epsilon = (1.0 - alpha) / ens_size

        var = torch.abs(preds.unsqueeze(dim=-1) - preds.unsqueeze(dim=-2))
        if antithetic_pairs:
            cross = antithetic_cross_pairs(ens_size, preds.device)
            mae = torch.abs(preds - targets.unsqueeze(dim=-1)).mean(dim=-1)
            return mae - 0.5 * (1 - epsilon) * (var * cross).sum(dim=(-1, -2)) / cross.sum()

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
        squash_mode: Squash_mode = "avg",
    ) -> torch.Tensor:
        is_sharded = grid_shard_slice is not None

        if self.ignore_nans:
            y_pred, y_target = self.mask_nans(y_pred, y_target)

        y_target = einops.rearrange(y_target, "bs t 1 latlon v -> bs t v latlon 1")
        y_pred = einops.rearrange(y_pred, "bs t e latlon v -> bs t v latlon e")

        context = (
            torch.amp.autocast(device_type=y_pred.device.type, enabled=False) if self.no_autocast else nullcontext()
        )
        with context:
            crps = self._kernel_crps(y_pred, y_target)

        crps = einops.rearrange(crps, "bs t v latlon -> bs t 1 latlon v")
        crps = self.scale(crps, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)

        return self.reduce(crps, squash=squash, squash_mode=squash_mode, group=group if is_sharded else None)

    @property
    def name(self) -> str:
        return f"crps{self.alpha:.2f}"
