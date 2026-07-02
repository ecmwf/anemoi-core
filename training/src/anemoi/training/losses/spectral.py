# (C) Copyright 2025-2026 Anemoi contributors.
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
from contextlib import nullcontext
from typing import TYPE_CHECKING
from typing import Literal

import einops
import torch

from anemoi.graphs.projection_helpers import DEFAULT_DATASET_NAME
from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.models.layers.sparse_projector import SparseProjector
from anemoi.models.layers.spectral_transforms import DCT2D
from anemoi.models.layers.spectral_transforms import FFT2D
from anemoi.models.layers.spectral_transforms import Cartesian2DTransform
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
    from torch_geometric.data import HeteroData

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
    needs_graph_data: bool = True

    def __init__(
        self,
        transform: Literal[
            "fft2d",
            "reduced_sht",
            "octahedral_sht",
            "dct2d",
        ] = "fft2d",
        *,
        subgrid: tuple[int, int | None] | None = None,
        projection_config: object | None = None,
        graph_data: HeteroData | None = None,
        data_node_name: str = DEFAULT_DATASET_NAME,
        ignore_nans: bool = False,
        scalers: list | None = None,
        **kwargs,
    ) -> None:
        """Create a spectral loss.

        Parameters
        ----------
        transform
            Spectral transform type.
        subgrid
            Optional ``(start, end)`` slice applied to the grid before the transform;
            ``end=None`` runs to the last gridpoint.
        projection_config
            Optional sparse-projection config applied after the slice and before the
            transform. See
            :meth:`~anemoi.models.layers.graph_provider.ProjectionGraphProvider.from_config`
            for the supported modes.
        graph_data
            Model graph; required when *projection_config* uses edge or target-grid mode.
        data_node_name
            Node type in *graph_data* holding the data-grid coordinates.
        ignore_nans
            Must be False; spectral losses cannot handle missing values.
        scalers
            Accepted for config backwards-compatibility; scaling is handled by BaseLoss.
        kwargs
            Forwarded to the spectral transform.
        """
        assert not ignore_nans, "Spectral losses cannot handle missing values; ignore_nans must be False"
        BaseLoss.__init__(self, ignore_nans=ignore_nans)

        # Backwards-compatibility: older configs pass scalers to the loss ctor.
        _ = scalers  # intentionally unused
        kwargs.pop("scalers", None)

        # Sharding over grid dimension is not supported for spectral transforms.
        # Enforce loss to be calculated on full grids.
        self.supports_sharding = False

        # subgrid selects a contiguous block of the grid before the transform. This only makes
        # sense for the Cartesian transforms (FFT2D/DCT2D); spherical harmonic transforms need the
        # whole domain to compute the spectra, so reject an explicit subgrid for them.
        if subgrid is not None and transform in ("reduced_sht", "octahedral_sht"):
            msg = (
                f"subgrid is not supported for the '{transform}' transform: "
                "spherical harmonic transforms require the full grid"
            )
            raise ValueError(msg)
        self.subgrid = slice(*(subgrid or (0, None)))
        self.projection_provider = ProjectionGraphProvider.from_config(
            projection_config,
            graph_data=graph_data,
            data_node_name=data_node_name,
        )
        if self.projection_provider is not None:
            self.projector = SparseProjector()

        if transform == "fft2d":
            LOGGER.info("Using FFT2D spectral transform in spectral loss.")
            self.transform = FFT2D(**kwargs)
        elif transform == "dct2d":
            LOGGER.info("Using DCT2D spectral transform in spectral loss.")
            self.transform = DCT2D(**kwargs)
        elif transform == "reduced_sht":
            # expected additional args: grid
            # optional args: truncation, use_graphed_rfft
            LOGGER.info("Using ReducedSHT spectral transform in spectral loss.")
            self.transform = ReducedSHT(**kwargs)
        elif transform == "octahedral_sht":
            # expected additional args: nlat
            # optional args: truncation, use_graphed_rfft
            LOGGER.info("Using Octahedral SHT spectral transform in spectral loss.")
            self.transform = OctahedralSHT(**kwargs)
        else:
            msg = f"Unknown transform type: {transform}"
            raise ValueError(msg)

    def _select_subgrid(self, x: torch.Tensor) -> torch.Tensor:
        # Obtain a subgrid by slicing the grid dim as a view, avoiding an explicit index-tensor allocation.
        index = [slice(None)] * x.ndim
        index[TensorDim.GRID] = self.subgrid
        return x[tuple(index)]

    def _select_and_project(self, x: torch.Tensor) -> torch.Tensor:
        x = self._select_subgrid(x)
        LOGGER.debug("Spectral loss: shape after subgrid selection: %s", tuple(x.shape))
        if self.projection_provider is not None:
            x = self.projector.project(x, self.projection_provider)
            LOGGER.debug("Spectral loss: shape after projection: %s", tuple(x.shape))
        return x

    def _to_spectral(self, x: torch.Tensor) -> torch.Tensor:
        """Select the node subset and optionally project to the target grid, then transform to the spectral domain."""
        return self.transform.forward(self._select_and_project(x))

    def _to_spectral_flat(self, x: torch.Tensor) -> torch.Tensor:
        """Transform to spectral domain and flatten the transformed dims into one "mode" axis."""
        x_spec = self._to_spectral(x)
        # the transform splits the single grid dim into two spectral dims; flatten them back to one
        return x_spec.flatten(start_dim=x_spec.ndim - 3, end_dim=-2)


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
    - ``fft2d`` / ``dct2d``: the ``(ky, kx)`` spectral plane is binned into integer
      radial-wavenumber bands :math:`L = \mathrm{round}(\sqrt{k_y^2 + k_x^2})`, and the
      sum is taken over each band. :math:`L` is then the radial wavenumber; the per-band
      terms have a slightly different physical meaning than under the SHT but the
      formulation is unchanged. Patch-wise FFT2D (``patch_size`` set) is not supported.
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
        # Patch-wise FFT2D yields a per-patch (ky, kx) plane that breaks the per-L contract.
        assert (
            getattr(self.transform, "patch_size", None) is None
        ), "SpectralAMSELoss does not support patch-wise FFT2D; set patch_size=None"
        # AMSE uses the Cartesian PSD/CSD path, which needs the radial-band index.
        if isinstance(self.transform, Cartesian2DTransform):
            self.transform._register_radial_bands()
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
            pred_spec = self._to_spectral(pred)
            target_spec = self._to_spectral(target)

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

        context = torch.amp.autocast(device_type=pred.device.type, enabled=False) if self.no_autocast else nullcontext()
        with context:
            # -> [..., modes, vars]
            pred_spec = self._to_spectral_flat(pred)
            tgt_spec = self._to_spectral_flat(target)

            pred_spec = einops.rearrange(pred_spec, "b t e m v -> b t v m e")  # ensemble dim last for preds
            tgt_spec = einops.rearrange(tgt_spec, "b t 1 m v -> b t v m 1")
            crps = self._kernel_crps(pred_spec, tgt_spec, alpha=self.alpha)

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
