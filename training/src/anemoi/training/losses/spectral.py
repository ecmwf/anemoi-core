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
  (i.e. `(..., grid, variables)`), and internally reshape for FFT2D or use
  the appropriate SHT for spherical grids.
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


def _resolve_spectral_without_scalers(
    scaler,
    without_scalers: list[str] | list[int] | None,
    *,
    keep_grid_scalers: set[str] | None = None,
) -> list[str]:
    """Resolve spectral loss scaler exclusions by scaler name.

    Spectral losses generally exclude grid-dimension scalers to avoid inconsistent
    weighting across transformed modes. This helper keeps that behavior while
    allowing selected grid scalers (e.g. ``qc_scaler``) to remain active.
    """
    keep_grid_scalers = keep_grid_scalers or set()
    excluded_names: set[str] = set()

    dims_to_exclude: list[int] = []
    if without_scalers is None or len(without_scalers) == 0:
        dims_to_exclude = [TensorDim.GRID.value]
    elif isinstance(without_scalers[0], str):
        excluded_names.update(str(name) for name in without_scalers)
    else:
        dims_to_exclude = [int(dim) for dim in without_scalers]
        if TensorDim.GRID.value not in dims_to_exclude:
            dims_to_exclude.append(TensorDim.GRID.value)

    if len(dims_to_exclude) > 0:
        for name, (dims, _tensor) in scaler.tensors.items():
            dims_tuple = tuple(int(d) for d in dims)
            if not any(dim in dims_tuple for dim in dims_to_exclude):
                continue
            if TensorDim.GRID.value in dims_tuple and name in keep_grid_scalers:
                continue
            excluded_names.add(name)

    return sorted(excluded_names)


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
        super().__init__(ignore_nans=ignore_nans)

        # Backwards-compatibility: older configs pass scalers to the loss ctor.
        _ = scalers  # intentionally unused
        kwargs.pop("scalers", None)
        x_dim = kwargs.get("x_dim")
        y_dim = kwargs.get("y_dim")

        # Sharding over grid dimension is not supported for spectral transforms.
        # Enforce loss to be calculated on full grids.
        self.supports_sharding = False

        if transform == "fft2d":
            LOGGER.info("Using FFT2D spectral transform in spectral loss.")
            self.transform = FFT2D(**kwargs)
            self.x_dim = int(x_dim) if x_dim is not None else None
            self.y_dim = int(y_dim) if y_dim is not None else None
        elif transform == "dct2d":
            LOGGER.info("Using DCT2D spectral transform in spectral loss.")
            self.transform = DCT2D(**kwargs)
            self.x_dim = int(x_dim) if x_dim is not None else None
            self.y_dim = int(y_dim) if y_dim is not None else None
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
        # flatten only transformed spatial/spectral dims into one "mode" axis
        spatial_start_dim = x.ndim - 2
        return x_spec.flatten(start_dim=spatial_start_dim, end_dim=-2)

    def _mask_nans_pre_transform(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask NaNs in spatial tensors before spectral transform."""
        if not self.ignore_nans:
            return pred, target

        target_nan_mask = torch.isnan(target)
        target = target.masked_fill(target_nan_mask, 0.0)

        pred_nan_mask = target_nan_mask
        while pred_nan_mask.ndim < pred.ndim:
            pred_nan_mask = pred_nan_mask.unsqueeze(TensorDim.ENSEMBLE_DIM.value)
        pred = pred.masked_fill(pred_nan_mask, 0.0)
        return pred, target


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
        del kwargs  # unused
        is_sharded = grid_shard_slice is not None
        group = group if is_sharded else None

        pred, target = self._mask_nans_pre_transform(pred, target)
        pred_spectral = self._to_spectral_flat(pred)
        target_spectral = self._to_spectral_flat(target)
        n_modes = pred_spectral.size(dim=TensorDim.GRID.value)

        diff = torch.abs(pred_spectral - target_spectral) ** 2

        result = self.scale(
            diff,
            scaler_indices,
            without_scalers=_resolve_spectral_without_scalers(
                self.scaler,
                without_scalers,
                keep_grid_scalers={"qc_scaler"},
            ),
            grid_shard_slice=grid_shard_slice,
        )
        result /= n_modes
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
        **kwargs,
    ) -> torch.Tensor:
        is_sharded = grid_shard_slice is not None
        group = group if is_sharded else None
        eps = torch.finfo(pred.dtype).eps

        pred, target = self._mask_nans_pre_transform(pred, target)
        pred_spectral = self._to_spectral_flat(pred)
        target_spectral = self._to_spectral_flat(target)
        n_modes = pred_spectral.size(dim=TensorDim.GRID.value)

        power_pred = torch.abs(pred_spectral) ** 2
        power_tgt = torch.abs(target_spectral) ** 2

        log_diff = torch.log(power_tgt + eps) - torch.log(power_pred + eps)

        result = self.scale(
            log_diff**2,
            scaler_indices,
            without_scalers=_resolve_spectral_without_scalers(
                self.scaler,
                without_scalers,
                keep_grid_scalers={"qc_scaler"},
            ),
            grid_shard_slice=grid_shard_slice,
        )
        result /= n_modes
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
        **kwargs,
    ) -> torch.Tensor:
        del kwargs  # unused
        is_sharded = grid_shard_slice is not None
        group = group if is_sharded else None
        eps = torch.finfo(pred.dtype).eps

        pred, target = self._mask_nans_pre_transform(pred, target)
        pred_spectral = self._to_spectral_flat(pred)
        target_spectral = self._to_spectral_flat(target)
        n_modes = pred_spectral.size(dim=TensorDim.GRID.value)

        # compute correlation per mode before applying any external weighting
        # keeps the ratio bounded by Cauchy-Schwarz (up to numerical error)
        cross = torch.real(pred_spectral * torch.conj(target_spectral))
        denom = torch.sqrt(torch.abs(pred_spectral) ** 2 * torch.abs(target_spectral) ** 2 + eps)
        correlation = torch.clamp(cross / denom, min=-1.0, max=1.0)

        # apply weighting/scaling after correlation is computed
        result = (1 - correlation) / n_modes
        result = self.scale(
            result,
            scaler_indices,
            without_scalers=_resolve_spectral_without_scalers(
                self.scaler,
                without_scalers,
                keep_grid_scalers={"qc_scaler"},
            ),
            grid_shard_slice=grid_shard_slice,
        )
        return self.reduce(result, squash=squash, group=group)


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
            alpha=alpha,
            no_autocast=no_autocast,
            ignore_nans=ignore_nans,
            scalers=scalers,
            **kwargs,
        )
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
        **kwargs,  # noqa: ARG002
    ) -> torch.Tensor:
        is_sharded = grid_shard_slice is not None
        group = group if is_sharded else None

        pred, target = self._mask_nans_pre_transform(pred, target)
        # → [..., modes, vars]
        pred_spec = self._to_spectral_flat(pred)
        tgt_spec = self._to_spectral_flat(target)
        n_modes = pred_spec.size(dim=TensorDim.GRID.value)

        pred_spec = einops.rearrange(pred_spec, "b t e m v -> b t v m e")  # ensemble dim last for preds
        tgt_spec = einops.rearrange(tgt_spec, "... m v -> (...) v m")  # remove ensemble dim for targets
        if self.no_autocast:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                crps = self._kernel_crps(pred_spec, tgt_spec, alpha=self.alpha)
        else:
            crps = self._kernel_crps(pred_spec, tgt_spec, alpha=self.alpha)
        crps = einops.rearrange(crps, "b t v m -> b t 1 m v")  # consistent with tensordim

        scaled = self.scale(
            crps,
            scaler_indices,
            without_scalers=_resolve_spectral_without_scalers(
                self.scaler,
                without_scalers,
                keep_grid_scalers={"qc_scaler"},
            ),
            grid_shard_slice=grid_shard_slice,
        )
        scaled /= n_modes
        return self.reduce(scaled, squash=squash, group=group)

    @property
    def name(self) -> str:
        return "CRPS-Spectral"
