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
from anemoi.training.losses.base import Squash_mode
from anemoi.training.losses.kcrps import CRPS
from anemoi.training.losses.kcrps import CRPSBackend
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
            Spectral losses cannot handle missing values;
            ignore_nans must be False.
        scalers
            Kept for Hydra/config backwards compatibility. This module does not
            consume this argument directly (scaling is handled by BaseLoss).
        kwargs
            Additional arguments for the spectral transform.
        """
        assert not ignore_nans, "Spectral losses cannot handle missing values; ignore_nans must be False"
        BaseLoss.__init__(self, ignore_nans=ignore_nans)

        # Backwards-compatibility: older configs pass scalers to the loss ctor.
        _ = scalers  # intentionally unused
        kwargs.pop("scalers", None)

        # Sharding over grid dimension is not supported for spectral transforms.
        # Enforce loss to be calculated on full grids.
        self.supports_sharding = False

        # Some transforms are proven to be linear, so we can compute the pred - target difference before taking the
        # transform for those. This reduces the number of required transforms from two to one
        self.diff_before_transform = False

        if transform == "fft2d":
            LOGGER.info("Using FFT2D spectral transform in spectral loss.")
            self.transform = FFT2D(**kwargs)
        elif transform == "dct2d":
            LOGGER.info("Using DCT2D spectral transform in spectral loss.")
            self.transform = DCT2D(**kwargs)
        elif transform == "reduced_sht":
            # expected additional args: grid
            # optional args: truncation
            LOGGER.info("Using ReducedSHT spectral transform in spectral loss.")
            self.transform = ReducedSHT(**kwargs)
            self.diff_before_transform = True
        elif transform == "octahedral_sht":
            # expected additional args: nlat
            # optional args: truncation
            LOGGER.info("Using Octahedral SHT spectral transform in spectral loss.")
            self.transform = OctahedralSHT(**kwargs)
            self.diff_before_transform = True
        else:
            msg = f"Unknown transform type: {transform}"
            raise ValueError(msg)

    def _to_spectral_flat(self, x: torch.Tensor) -> torch.Tensor:
        """Transform to spectral domain and flatten spectral dimensions."""
        x_spec = self.transform.forward(x)
        # flatten only transformed spatial/spectral dims into one "mode" axis
        spatial_start_dim = x.ndim - 2
        return x_spec.flatten(start_dim=spatial_start_dim, end_dim=-2)


class SpectralAMSELoss(SpectralLoss):
    r"""Adjusted Mean Squared Error (AMSE) loss in spectral domain.

    Implements the AMSE formula from Subich et al. (arXiv:2501.19374, 2025):

    .. math::

        \text{AMSE} = \sum_L \left[
            \left( \sqrt{S^\text{pred}_L} - \sqrt{S^\text{target}_L} \right)^2
            + 2 \max\!\left(S^\text{pred}_L,\, S^\text{target}_L\right)
              \left(1 - \gamma_L \right)
        \right]

    where

    .. math::

        S_L = \sum_M \left| c_{L,M} \right|^2, \qquad
        \gamma_L = \frac{
            \operatorname{Re}\!\left[\sum_M c^\text{pred}_{L,M}
                \overline{c^\text{target}_{L,M}}\right]
        }{
            \sqrt{S^\text{pred}_L}\,\sqrt{S^\text{target}_L} + \varepsilon
        }.

    The sum over :math:`M` is performed before the nonlinear AMSE computation.

    The physical interpretation of :math:`L` and :math:`M` depends on the
    spectral transform:

    - ``octahedral_sht`` / ``reduced_sht``: :math:`L` is the total wavenumber
      and :math:`M` is the zonal wavenumber, consistent with the original paper.
      The sum over :math:`M` gives per-total-wavenumber power spectra.
    - ``fft2d`` / ``dct2d``: currently not supported. These transforms require
      implementations of ``power_spectral_density()`` and
      ``cross_spectral_density()`` compatible with the AMSE formulation.
    """

    def __init__(self, *args, eps: float = 1e-8, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # assert if PSD is defined for the transform, since AMSE relies on it
        assert hasattr(self.transform, "power_spectral_density") and callable(
            self.transform.power_spectral_density,
        ), "spectral transform used in SpectralAdjustedMeanSquaredError must contain a PSD method"
        assert hasattr(self.transform, "cross_spectral_density") and callable(
            self.transform.cross_spectral_density,
        ), "spectral transform used in SpectralAdjustedMeanSquaredError must contain a cross-spectrum method"
        self.eps = eps

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
        squash_mode: str = "avg",
        **kwargs,
    ) -> torch.Tensor:
        del kwargs  # unused
        is_sharded = grid_shard_slice is not None
        group = group if is_sharded else None

        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            # transform to spectral domain: [B, T, E, grid, vars] -> [B, T, E, L, M, vars]
            # don't flatten to modes here since we need to calculate PSD and coherence per-L
            pred_spec = self.transform.forward(pred)
            target_spec = self.transform.forward(target)

            # per-L PSD: [B, T, E, L, vars]
            psd_pred = self.transform.power_spectral_density(pred_spec)
            psd_target = self.transform.power_spectral_density(target_spec)
            # cross-spectrum summed over M: [B, T, E, L, vars]
            cross = self.transform.cross_spectral_density(pred_spec, target_spec)

            amp_pred = torch.sqrt(psd_pred + self.eps)
            amp_target = torch.sqrt(psd_target + self.eps)
            coherence = cross / (amp_pred * amp_target + self.eps)

            # per-L AMSE: [B, T, E, L, vars]
            amse_per_l = (amp_pred - amp_target) ** 2 + 2 * torch.maximum(psd_pred, psd_target) * (1 - coherence)

        result = self.scale(
            amse_per_l,
            scaler_indices,
            without_scalers=_ensure_without_scalers_has_grid_dimension(without_scalers),
            grid_shard_slice=grid_shard_slice,
        )
        return self.reduce(result, squash=squash, group=group, squash_mode=squash_mode)


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
        squash_mode: Squash_mode = "avg",
        **kwargs,
    ) -> torch.Tensor:
        del kwargs  # unused
        is_sharded = grid_shard_slice is not None
        group = group if is_sharded else None

        # If we know for sure this transform is linear, we can compute the residual before taking the transform
        if self.diff_before_transform:
            diff = torch.abs(self._to_spectral_flat(pred - target)) ** 2
        else:
            pred_spectral = self._to_spectral_flat(pred)
            target_spectral = self._to_spectral_flat(target)
            diff = torch.abs(pred_spectral - target_spectral) ** 2

        result = self.scale(
            diff,
            scaler_indices,
            without_scalers=_ensure_without_scalers_has_grid_dimension(without_scalers),
            grid_shard_slice=grid_shard_slice,
        )
        return self.reduce(result, squash=squash, group=group, squash_mode=squash_mode)


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
        squash_mode: Squash_mode = "avg",
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
        return torch.sqrt(self.reduce(result, squash=squash, group=group, squash_mode=squash_mode) + eps)


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
        squash_mode: Squash_mode = "avg",
    ) -> torch.Tensor:
        is_sharded = grid_shard_slice is not None
        group = group if is_sharded else None
        eps = torch.finfo(pred.dtype).eps

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
            without_scalers=_ensure_without_scalers_has_grid_dimension(without_scalers),
            grid_shard_slice=grid_shard_slice,
        )
        return self.reduce(result, squash=squash, group=group, squash_mode=squash_mode)


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


class SpectralCRPSLoss(SpectralLoss, CRPS):
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
        backend: CRPSBackend = "stable",
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
        self._validate_arguments(alpha, backend)
        self.alpha = alpha
        self.backend = backend
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
        squash_mode: Squash_mode = "avg",
    ) -> torch.Tensor:
        is_sharded = grid_shard_slice is not None
        group = group if is_sharded else None

        # → [..., modes, vars]
        pred_spec = self._to_spectral_flat(pred)
        tgt_spec = self._to_spectral_flat(target)

        pred_spec = einops.rearrange(pred_spec, "b t e m v -> b t v m e")  # ensemble dim last for preds
        tgt_spec = einops.rearrange(tgt_spec, "... m v -> (...) v m")  # remove ensemble dim for targets
        if self.no_autocast:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                crps = self._kernel_crps(pred_spec, tgt_spec)
        else:
            crps = self._kernel_crps(pred_spec, tgt_spec)
        crps = einops.rearrange(crps, "b t v m -> b t 1 m v")  # consistent with tensordim

        scaled = self.scale(
            crps,
            scaler_indices,
            without_scalers=_ensure_without_scalers_has_grid_dimension(without_scalers),
            grid_shard_slice=grid_shard_slice,
        )
        return self.reduce(scaled, squash=squash, group=group, squash_mode=squash_mode)

    @property
    def name(self) -> str:
        return "CRPS-Spectral"
