# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

r"""Spectral smoothness penalty on the model increment.

Unlike :class:`~anemoi.training.losses.GraphLaplacianSmoothnessLoss`, which penalises the
roughness of the full predicted field (and so also damps real troughs and ridges), this loss
penalises the roughness of the *increment* ``delta = pred - reference``, where ``reference``
is the residual base the model added its output to: the previous state for a forecast step,
the model background for a DA step.

The penalty is the area mean of ``(length_scale^(2p) nabla^(2p) delta)^2`` with ``p = order``
(1 = Laplacian, 2 = biharmonic), evaluated exactly in spherical-harmonic space via Parseval:

.. math::
    \mathcal{P} = \sum_l \Bigl(l(l+1)\,\frac{\ell^2}{R^2}\Bigr)^{2p} E_l,
    \qquad E_l = \frac{|c_{l0}|^2 + 2\sum_{m>0}|c_{lm}|^2}{|c_{00}[1]|^2},

where ``E_l`` is the share of the area-mean square of ``delta`` in total wavenumber ``l``.
A biharmonic penalty weights wavenumber ``l`` by ``~l^8``: synoptic increments cost almost
nothing, grid-scale wiggles a lot. Because ``l`` is a physical wavenumber, a given loss weight
means the same thing at O96 and N320. (A biharmonic built from repeated graph Laplacians is
not usable here: on irregular stencils the first Laplacian's truncation error is grid-scale
noise that the second one amplifies by ``1/h^2``.)

The training method supplies ``reference`` (prediction layout) through the loss call. A missing
``reference`` kwarg raises; an explicit ``reference=None`` means "no background exists for this
step" (the DA cold start) and contributes zero.

Usage, inside a CombinedLoss (no grid scalers: the spectral sum is already area-weighted):

.. code-block:: yaml

    - _target_: anemoi.training.losses.SpectralIncrementSmoothnessLoss
      transform: octahedral_sht
      nlat: 192            # O96; for N320 use transform: reduced_sht, grid: n320
      order: 2
      length_scale_km: 100.0
      predicted_variables: [z_500, msl]
      scalers: []
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

import torch

from anemoi.training.losses.spectral import SpectralLoss
from anemoi.training.losses.spectral import _assert_spectral_scalers_compatible
from anemoi.training.utils.enums import TensorDim

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

    from anemoi.models.distributed.shapes import ShardSizes
    from anemoi.training.losses.base import Squash_mode

LOGGER = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0

# Sentinel distinguishing "the method never passed a reference" (a wiring error) from an
# explicit ``reference=None`` (no background for this step).
_MISSING = object()


class SpectralIncrementSmoothnessLoss(SpectralLoss):
    """Area-mean ``(nabla^(2*order) (pred - reference))^2`` via a spherical-harmonic transform."""

    # Tells the training method to pass ``reference`` (the residual base) into the loss call.
    needs_reference: bool = True

    def __init__(
        self,
        transform: Literal["reduced_sht", "octahedral_sht"] = "octahedral_sht",
        *,
        order: int = 2,
        length_scale_km: float = 100.0,
        min_wavenumber: int = 0,
        penalty_weight: float = 1.0,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        """Initialise the increment smoothness loss.

        Parameters
        ----------
        transform : {"octahedral_sht", "reduced_sht"}
            Spherical-harmonic transform. ``octahedral_sht`` needs ``nlat`` (192 for O96);
            ``reduced_sht`` needs ``grid`` (``n320``). ``truncation`` is optional.
        order : int
            Number of Laplacians: 1 = Laplacian, 2 = biharmonic.
        length_scale_km : float
            Length unit of the operator; the penalty is in units of ``length_scale^(4*order)``.
        min_wavenumber : int
            Total wavenumbers ``l < min_wavenumber`` are not penalised (high-pass). The ``~l^8``
            weight already makes them cheap but not free; with a cutoff (e.g. 20-30 at O96) the
            penalty cannot trade grid-scale roughness against planetary-scale / global-mean drift.
            0 (default) penalises every wavenumber.
        penalty_weight : float
            Multiplier on the penalty (on top of the CombinedLoss weight).
        ignore_nans : bool
            Must be False: pred and reference carry no NaNs (spectral transforms cannot handle them).
        **kwargs
            Forwarded to :class:`SpectralLoss` (``nlat``/``grid``/``truncation``, graph data).
        """
        if transform not in ("reduced_sht", "octahedral_sht"):
            msg = f"{type(self).__name__} needs a spherical-harmonic transform, got {transform!r}."
            raise ValueError(msg)
        if order < 1:
            msg = f"order must be >= 1, got {order}"
            raise ValueError(msg)
        if length_scale_km <= 0:
            msg = f"length_scale_km must be > 0, got {length_scale_km}"
            raise ValueError(msg)
        if min_wavenumber < 0:
            msg = f"min_wavenumber must be >= 0, got {min_wavenumber}"
            raise ValueError(msg)
        super().__init__(transform=transform, ignore_nans=ignore_nans, **kwargs)
        self.order = int(order)
        self.length_scale_km = float(length_scale_km)
        self.penalty_weight = penalty_weight
        self._reference_checked = False

        # Weight per (l, m): Parseval normalisation (|c_00|^2 of a unit field, m>0 counted twice
        # for a real field) times the operator's squared eigenvalue (l(l+1) (ls/R)^2)^(2 order).
        n_points = self.transform._sht.n_grid_points
        unit = self.transform(torch.ones(1, 1, 1, n_points, 1))[0, 0, 0, ..., 0]
        n_l, n_m = unit.shape
        norm = float(unit[0, 0].real ** 2 + unit[0, 0].imag ** 2)
        ell = torch.arange(n_l, dtype=torch.float64)
        eigen = (ell * (ell + 1) * (self.length_scale_km / EARTH_RADIUS_KM) ** 2) ** (2 * self.order)
        m_factor = torch.full((n_m,), 2.0, dtype=torch.float64)
        m_factor[0] = 1.0
        if min_wavenumber >= n_l:
            msg = f"min_wavenumber {min_wavenumber} >= number of resolved wavenumbers {n_l}: penalty would be zero."
            raise ValueError(msg)
        self.min_wavenumber = int(min_wavenumber)
        eigen[: self.min_wavenumber] = 0.0
        weight = eigen.view(-1, 1) * m_factor.view(1, -1) / norm
        self.register_buffer("mode_weight", weight.float(), persistent=False)
        LOGGER.info(
            "%s: %s, %d grid points, lmax %d, order %d, length scale %.0f km, min wavenumber %d.",
            type(self).__name__,
            transform,
            n_points,
            n_l - 1,
            self.order,
            self.length_scale_km,
            self.min_wavenumber,
        )

    @property
    def name(self) -> str:
        return "spectral_increment_smoothness"

    def spectrum(self, delta: torch.Tensor) -> torch.Tensor:
        """Weighted energy per total wavenumber, shape (bs, time, ens, L, vars)."""
        coeffs = self._to_spectral(delta)  # (bs, time, ens, L, M, vars), complex
        power = coeffs.real.square() + coeffs.imag.square()
        return (power * self.mode_weight.unsqueeze(-1)).sum(dim=-2)

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
        grid_shard_sizes: ShardSizes = None,
        grid_dim: int | None = None,
        squash_mode: Squash_mode = "avg",
        reference: Any = _MISSING,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the increment smoothness penalty.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction, shape (bs, time, ens, grid, vars).
        target : torch.Tensor
            Unused: this penalty acts on the increment only.
        squash : bool, optional
            Average over the variable dimension, by default True.
        scaler_indices : tuple[int, ...] | None, optional
            Scaler indices for loss scaling.
        without_scalers : list[str] | list[int] | None, optional
            Scalers to exclude.
        grid_shard_slice : slice | None, optional
            Grid shard slice when the inputs are grid-sharded.
        group : ProcessGroup | None, optional
            Process group for distributed operation.
        grid_shard_sizes : ShardSizes, optional
            Grid shard sizes when the inputs are grid-sharded.
        grid_dim : int | None, optional
            Grid dimension of the inputs, by default ``TensorDim.GRID``.
        squash_mode : Squash_mode, optional
            Reduction over the variable dimension, by default "avg".
        reference : torch.Tensor | None
            Residual base in the prediction's layout and shape. ``None`` (no background, e.g.
            the DA cold start) contributes zero; omitting the kwarg raises.
        **kwargs
            Ignored.

        Returns
        -------
        torch.Tensor
            Penalty, a scalar if ``squash`` else one value per variable.
        """
        del target, kwargs
        if reference is _MISSING:
            msg = (
                f"{type(self).__name__} needs the training method to pass `reference` (the residual "
                "base) to the loss; this method does not supply one."
            )
            raise RuntimeError(msg)
        if reference is None:
            return pred.sum() * 0.0
        if reference.shape != pred.shape:
            msg = f"reference shape {tuple(reference.shape)} must match pred shape {tuple(pred.shape)}."
            raise ValueError(msg)
        if not self._reference_checked:
            # Once: a non-finite reference means a selected variable has no residual base
            # (e.g. a diagnostic), which is a config error.
            if not torch.isfinite(reference).all():
                msg = (
                    f"{type(self).__name__}: reference has non-finite values. Only prognostic "
                    "variables have a residual base; restrict predicted_variables to prognostics."
                )
                raise ValueError(msg)
            self._reference_checked = True

        grid_dim = TensorDim.GRID if grid_dim is None else grid_dim
        # Grid-sharded inputs are moved to a full-grid, channel-sharded layout for the transform.
        pred, reference, channel_shard_sizes = self._prepare_for_spectral_transform(
            pred,
            reference,
            group,
            grid_shard_slice,
            grid_shard_sizes,
            grid_dim,
        )
        is_sharded = channel_shard_sizes is not None
        group = group if is_sharded else None

        # float32 with autocast off: the transform and the l^(4 order) weights are not bf16-safe, and under
        # bf16-mixed autocast the SHT einsum would otherwise downcast back to bf16 (view_as_complex rejects it).
        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            out = self.penalty_weight * self.spectrum(pred.float() - reference.float())

        out, spectral_shard_slice, global_spectral_size = self._reshard_spectral_loss(
            out,
            group,
            channel_shard_sizes,
            grid_dim,
        )
        _assert_spectral_scalers_compatible(self.scaler, global_spectral_size)
        out = self.scale(out, scaler_indices, without_scalers=without_scalers, grid_shard_slice=spectral_shard_slice)
        # reduce() sums over the grid (here: wavenumber) dimension, as Parseval requires.
        return self.reduce(out, squash=squash, group=group, squash_mode=squash_mode)
