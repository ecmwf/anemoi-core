# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging

import einops
import torch

from anemoi.training.losses.weightedloss import BaseWeightedLoss

LOGGER = logging.getLogger(__name__)


class KernelCRPS(BaseWeightedLoss):
    """Area-weighted kernel CRPS loss."""

    def __init__(
        self,
        node_weights: torch.Tensor,
        fair: bool = True,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        """Latitude- and (inverse-)variance-weighted kernel CRPS loss.

        Args:
        node_weights : torch.Tensor
            Weights by area
        fair: calculate a "fair" (unbiased) score - ensemble variance component weighted by (ens-size-1)^-1
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False
        """
        super().__init__(node_weights=node_weights, ignore_nans=ignore_nans, **kwargs)

        self.fair = fair

    def _kernel_crps(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Kernel (ensemble) CRPS.

        Args:
            preds: predicted ensemble, shape (batch_size, n_vars, latlon, ens_size)
            targets: ground truth, shape (batch_size, n_vars, latlon)

        Returns
        -------
        kCRPS value
            The point-wise kernel CRPS, shape (batch_size, 1, latlon).
        """
        ens_size = preds.shape[-1]
        mae = torch.mean(torch.abs(targets[..., None] - preds), dim=-1)

        assert ens_size > 1, "Ensemble size must be greater than 1."

        coef = -1.0 / (ens_size * (ens_size - 1)) if self.fair else -1.0 / (ens_size**2)

        ens_var = torch.zeros(size=preds.shape[:-1], device=preds.device)
        for i in range(ens_size):  # loop version to reduce memory usage
            ens_var += torch.sum(torch.abs(preds[..., i].unsqueeze(-1) - preds[..., i + 1 :]), dim=-1)
        ens_var = coef * ens_var

        return mae + ens_var

    def forward(
        self,
        y_pred: torch.Tensor,
        y_target: torch.Tensor,
        squash: bool = True,
        *,
        scalar_indices: tuple[int, ...] | None = None,
        without_scalars: list[str] | list[int] | None = None,
    ) -> torch.Tensor:

        y_target = einops.rearrange(y_target, "bs latlon v -> bs v latlon")
        y_pred = einops.rearrange(y_pred, "bs e latlon v -> bs v latlon e")

        bs_ = y_pred.shape[0]  # batch size
        kcrps_ = self._kernel_crps(y_pred, y_target)

        kcrps_ = einops.rearrange(kcrps_, "bs v latlon -> bs latlon v")

        kcrps_ = self.scale(kcrps_, scalar_indices, without_scalars=without_scalars)
        kcrps_ = kcrps_ * self.node_weights[:, None]

        # divide by (weighted point count) * (batch size)
        npoints = torch.sum(self.node_weights)
        if squash:
            return kcrps_.sum() / (npoints * bs_)
        # sum only across the batch dimension; enable this to generate per-variable CRPS "maps"
        loss = kcrps_.sum(dim=0) / bs_
        return loss.sum(dim=0) / self.node_weights.sum()

    @property
    def name(self) -> str:

        f_str = "f" if self.fair else ""

        return f"{f_str}kcrps"


class AlmostFairKernelCRPS(BaseWeightedLoss):
    """Area-weighted almost fair kernel CRPS loss."""

    def __init__(
        self,
        node_weights: torch.Tensor,
        alpha: float = 1.0,
        no_autocast: bool = True,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        """Latitude- and (inverse-)variance-weighted kernel CRPS loss.

        Args:
        node_weights : torch.Tensor
            Weights by area
        alpha: factor for linear combination of fair (unbiased, ensemble variance component weighted by (ens-size-1)^-1)
            and standard CRPS (1.0 = fully fair, 0.0 = fully unfair)
        no_autocast: deactivate autocast for the kernel CRPS calculation
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False
        """
        super().__init__(node_weights=node_weights, ignore_nans=ignore_nans, **kwargs)

        self.alpha = alpha
        self.no_autocast = no_autocast

    def _kernel_crps(self, preds: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Kernel (ensemble) CRPS.

        Args:
            preds: predicted ensemble, shape (batch_size, n_vars, latlon, ens_size)
            targets: ground truth, shape (batch_size, n_vars, latlon)
            alpha: factor for linear combination of fair
                (unbiased, ensemble variance component weighted by (ens-size-1)^-1)
                and standard CRPS (1.0 = fully fair, 0.0 = fully unfair)

        Returns
        -------
        kCRPS value
            The point-wise kernel CRPS, shape (batch_size, 1, latlon).
        """
        ens_size = preds.shape[-1]

        epsilon = (1.0 - alpha) / ens_size

        var = torch.abs(preds.unsqueeze(dim=-1) - preds.unsqueeze(dim=-2))
        diag = torch.eye(ens_size, dtype=torch.bool, device=preds.device)
        err_r = einops.repeat(
            torch.abs(preds - targets.unsqueeze(dim=-1)),
            "batch var latlon ens -> batch var latlon n ens",
            n=ens_size,
        )

        mem_err = err_r * ~diag
        mem_err_transpose = mem_err.transpose(-1, -2)

        assert ens_size > 1, "Ensemble size must be greater than 1."

        coef = 1.0 / (2.0 * ens_size * (ens_size - 1))
        return coef * torch.sum(mem_err + mem_err_transpose - (1 - epsilon) * var, dim=(-1, -2))

    def forward(
        self,
        y_pred: torch.Tensor,
        y_target: torch.Tensor,
        squash: bool = True,
        *,
        scalar_indices: tuple[int, ...] | None = None,
        without_scalars: list[str] | list[int] | None = None,
    ) -> torch.Tensor:

        y_target = einops.rearrange(y_target, "bs latlon v -> bs v latlon")
        y_pred = einops.rearrange(y_pred, "bs e latlon v -> bs v latlon e")
        bs_ = y_pred.shape[0]  # batch size

        if self.no_autocast:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                kcrps_ = self._kernel_crps(y_pred, y_target, alpha=self.alpha)
        else:
            kcrps_ = self._kernel_crps(y_pred, y_target, alpha=self.alpha)

        kcrps_ = einops.rearrange(kcrps_, "bs v latlon -> bs latlon v")

        kcrps_ = self.scale(kcrps_, scalar_indices, without_scalars=without_scalars)
        kcrps_ = kcrps_ * self.node_weights[:, None]

        # divide by (weighted point count) * (batch size)
        npoints = torch.sum(self.node_weights)
        if squash:
            return kcrps_.sum() / (npoints * bs_)

        # sum only across the batch dimension; enable this to generate per-variable CRPS "maps"
        loss = kcrps_.sum(dim=0) / bs_
        return loss.sum(dim=0) / self.node_weights.sum()

    @property
    def name(self) -> str:
        return f"afkcrps{self.alpha:.2f}"
