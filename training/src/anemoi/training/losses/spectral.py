# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Spectral-domain losses.

This module consolidates spectral losses that were historically split across
`spatial.py` and `spectral.py`.

Notes
-----
* These losses operate on tensors whose *spatial* dimension is flattened
  (i.e. `(..., grid, variables)`), and internally reshape to 2D grids for FFT2D.
* For backwards compatibility, legacy class names (e.g. ``LogFFT2Distance``)
  are kept.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Literal

import einops
import torch

from anemoi.models.layers.spectral_transforms import DCT2D
from anemoi.models.layers.spectral_transforms import FFT2D
from anemoi.models.layers.spectral_transforms import OctahedralSHT
from anemoi.models.layers.spectral_transforms import ReducedSHT
from anemoi.models.layers.spectral_transforms import SpectralTransform
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.kcrps import AlmostFairKernelCRPS
from anemoi.training.utils.enums import TensorDim

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

LOGGER = logging.getLogger(__name__)


def _ensure_without_scalers_has_grid_dimension(without_scalers: list[str] | list[int] | None) -> list[str] | list[int]:
    """Temporary fix for https://github.com/ecmwf/anemoi-core/issues/725.

    Some pipelines pass numeric scaler indices and rely on excluding scalers over grid dimension
    by default. Ensure this exclusion is present for numeric lists.
    """
    if without_scalers is None:
        return [TensorDim.GRID.value]
    if len(without_scalers) == 0:
        return [TensorDim.GRID.value]
    if not isinstance(without_scalers[0], str) and TensorDim.GRID.value not in without_scalers:
        without_scalers.append(TensorDim.GRID.value)  # type: ignore[arg-type]
    return without_scalers


class SpectralLoss(BaseLoss):
    """Base class for spectral losses."""

    transform: SpectralTransform

    def __init__(
        self,
        transform: Literal[
            "fft2d",
            "reduced_sht",
            "octahedral_sht",
            "dct2d",
        ] = "fft2d",
        *,
        ignore_nans: bool = False,
        scalers: list | None = None,
        **kwargs,
    ) -> None:
        """Create a spectral loss.

        Parameters
        ----------
        transform
            Spectral transform type.
        ignore_nans
            Whether to ignore NaNs in the loss computation.
        scalers
            Kept for Hydra/config backwards compatibility. This module does not
            consume this argument directly (scaling is handled by BaseLoss).
        kwargs
            Additional arguments for the spectral transform.
        """
        super().__init__(ignore_nans)

        # Backwards-compatibility: older configs pass scalers to the loss ctor.
        _ = scalers  # intentionally unused
        kwargs.pop("scalers", None)

        # Sharding over grid dimension is not supported for spectral transforms.
        # Enforce loss to be calculated on full grids.
        self.supports_sharding = False

        if transform == "fft2d":
            LOGGER.info("Using FFT2D spectral transform in spectral loss.")
            self.transform = FFT2D(**kwargs)
        elif transform == "dct2d":
            LOGGER.info("Using DCT2D spectral transform in spectral loss.")
            self.transform = DCT2D(**kwargs)
        elif transform == "reduced_sht":
            # expected additional args: grid
            LOGGER.info("Using ReducedSHT spectral transform in spectral loss.")
            self.transform = ReducedSHT(**kwargs)
        elif transform == "octahedral_sht":
            # expected additional args: lmax/mmax/folding
            LOGGER.info("Using Octahedral SHT spectral transform in spectral loss.")
            self.transform = OctahedralSHT(**kwargs)
        else:
            msg = f"Unknown transform type: {transform}"
            raise ValueError(msg)

    def _to_spectral_flat(self, x: torch.Tensor) -> torch.Tensor:
        """Transform to spectral domain and flatten spectral dimensions."""
        x_spec = self.transform.forward(x)
        return einops.rearrange(x_spec, "... y x v -> ... (y x) v")


class SpectralL2Loss(SpectralLoss):
    r"""L2 loss in spectral domain.

    .. math::
        \lVert F - \hat F \rVert_2^2
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs  # unused
        is_sharded = grid_shard_slice is not None
        group = group if is_sharded else None

        pred_spectral = self._to_spectral_flat(pred)
        target_spectral = self._to_spectral_flat(target)

        diff = torch.abs(pred_spectral - target_spectral) ** 2

        result = self.scale(
            diff,
            scaler_indices,
            without_scalers=_ensure_without_scalers_has_grid_dimension(without_scalers),
            grid_shard_slice=grid_shard_slice,
        )
        return self.reduce(result, squash=squash, group=group)


class LogSpectralDistance(SpectralLoss):
    r"""Log Spectral Distance (LSD)."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
    ) -> torch.Tensor:
        is_sharded = grid_shard_slice is not None
        group = group if is_sharded else None
        eps = torch.finfo(pred.dtype).eps

        pred_spectral = self._to_spectral_flat(pred)
        target_spectral = self._to_spectral_flat(target)

        power_pred = torch.abs(pred_spectral) ** 2
        power_tgt = torch.abs(target_spectral) ** 2

        log_diff = torch.log(power_tgt + eps) - torch.log(power_pred + eps)

        result = self.scale(
            log_diff**2,
            scaler_indices,
            without_scalers=_ensure_without_scalers_has_grid_dimension(without_scalers),
            grid_shard_slice=grid_shard_slice,
        )
        return torch.sqrt(self.reduce(result, squash=squash, group=group) + eps)


class FourierCorrelationLoss(SpectralLoss):
    r"""Fourier Correlation Loss (FCL)."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
    ) -> torch.Tensor:
        is_sharded = grid_shard_slice is not None
        group = group if is_sharded else None
        eps = torch.finfo(pred.dtype).eps

        pred_spectral = self._to_spectral_flat(pred)
        target_spectral = self._to_spectral_flat(target)

        cross = torch.real(pred_spectral * torch.conj(target_spectral))

        cross = self.scale(
            cross,
            scaler_indices,
            without_scalers=_ensure_without_scalers_has_grid_dimension(without_scalers),
            grid_shard_slice=grid_shard_slice,
        )
        numerator = 0.5 * torch.sum(cross, dim=TensorDim.GRID.value, keepdim=True)

        denom = torch.sqrt(
            torch.sum(torch.abs(pred_spectral) ** 2, dim=TensorDim.GRID.value, keepdim=True)
            * torch.sum(torch.abs(target_spectral) ** 2, dim=TensorDim.GRID.value, keepdim=True)
            + eps,
        )

        return self.reduce(1 - numerator / denom, squash=squash, group=group)


class LogFFT2Distance(LogSpectralDistance):
    """Backwards compatible alias for log spectral distance on FFT2D grids."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        ignore_nans: bool = False,
        scalers: list | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            transform="fft2d",
            x_dim=x_dim,
            y_dim=y_dim,
            ignore_nans=ignore_nans,
            scalers=scalers,
            **kwargs,
        )


class SpectralCRPSLoss(SpectralLoss, AlmostFairKernelCRPS):
    """CRPS computed in spectral space using arbitrary spectral transforms.

    Works with:
      - FFT2D
      - DCT2D
      - Reduced SHT
      - Octahedral SHT
    """

    def __init__(
        self,
        transform: Literal[
            "fft2d",
            "dct2d",
            "reduced_sht",
            "octahedral_sht",
        ] = "fft2d",
        *,
        x_dim: int | None = None,
        y_dim: int | None = None,
        alpha: float = 1.0,
        no_autocast: bool = True,
        ignore_nans: bool = False,
        scalers: list | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            transform=transform,
            x_dim=x_dim,
            y_dim=y_dim,
            ignore_nans=ignore_nans,
            scalers=scalers,
            **kwargs,
        )
        self.alpha = alpha
        self.no_autocast = no_autocast

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        **kwargs,
    ) -> torch.Tensor:
        is_sharded = grid_shard_slice is not None
        group = group if is_sharded else None

        # → [..., modes, vars]
        pred_spec = self._to_spectral_flat(pred)
        tgt_spec = self._to_spectral_flat(target)

        pred_spec = einops.rearrange(pred_spec, "b e m v -> b v m e")  # ensemble dim last for preds
        tgt_spec = einops.rearrange(tgt_spec, "... m v -> (...) v m")  # remove ensemble dim for targets
        if self.no_autocast:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                crps = self._kernel_crps(pred_spec, tgt_spec, alpha=self.alpha)
        else:
            crps = self._kernel_crps(pred_spec, tgt_spec, alpha=self.alpha)
        crps = einops.rearrange(crps, "b v m -> b 1 m v")  # consistent with tensordim

        scaled = self.scale(
            crps,
            scaler_indices,
            without_scalers=_ensure_without_scalers_has_grid_dimension(without_scalers),
            grid_shard_slice=grid_shard_slice,
        )
        return self.reduce(scaled, squash=squash, group=group)

    @property
    def name(self) -> str:
        return "CRPS-Spectral"
