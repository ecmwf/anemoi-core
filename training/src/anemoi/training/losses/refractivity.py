# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""GNSS-RO refractivity observation operator and loss.

The operator maps a predicted pressure-level column (geopotential ``z_<p>`` in
m^2 s^-2, temperature ``t_<p>`` in K, specific humidity ``q_<p>`` in kg/kg) to
refractivity at fixed geopotential heights via the Smith-Weintraub formula

    N = 77.6 p/T + 3.73e5 e/T^2      (p, e in hPa)

and the loss penalises ``ln N_model - ln N_obs`` against NaN-masked ROM SAF
refractivity columns (``refrac_<height_gpm>``) that are declared as ``target``
variables in the data config.
"""

from __future__ import annotations

import logging
from itertools import pairwise
from typing import TYPE_CHECKING
from typing import Literal

import torch

from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import LossFactoryContextKey
from anemoi.training.losses.base import Squash_mode
from anemoi.training.utils.index_space import IndexSpace

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

    from anemoi.models.data_indices.collection import IndexCollection

LOGGER = logging.getLogger(__name__)

# Smith-Weintraub refractivity constants (Smith & Weintraub 1953; Bevis et al. 1994 rounded).
K1_DRY: float = 77.6  # K hPa^-1
K2_WET: float = 3.73e5  # K^2 hPa^-1
# Ratio of gas constants dry air / water vapour, used for the vapour-pressure conversion
# e = q p / (EPS + (1 - EPS) q).
EPS_RD_RV: float = 0.622
# Standard gravity, converts geopotential metres to geopotential (m^2 s^-2).
G0: float = 9.80665

InterpMethod = Literal["linear_lnp", "hydrostatic_shape"]
HumidityInterp = Literal["linear", "log"]
# Floor applied before taking logs of specific humidity (kg/kg).
Q_FLOOR: float = 1e-9


def bracket_column(
    phi: torch.Tensor,
    phi_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Locate the model layer bracketing each target geopotential.

    Parameters
    ----------
    phi : torch.Tensor
        Geopotential ladder, shape ``(..., L)``. Expected strictly ascending along
        the last axis (pressure descending) but not assumed: columns that are not
        monotone, and targets matched by zero or several layers, are flagged invalid.
    phi_target : torch.Tensor
        Target geopotentials, shape ``(H,)``.

    Returns
    -------
    layer : torch.Tensor
        Long tensor ``(..., H)`` with the lower-level index of the bracketing layer
        (0 where invalid).
    weight : torch.Tensor
        Interpolation weight ``(..., H)`` in ``[0, 1]`` (0 where invalid).
    valid : torch.Tensor
        Bool tensor ``(..., H)``: exactly one layer brackets the target.
    """
    lo = phi[..., :-1].unsqueeze(-2)  # (..., 1, L-1)
    hi = phi[..., 1:].unsqueeze(-2)
    target = phi_target.reshape((1,) * (phi.dim() - 1) + (-1, 1))  # (1..., H, 1)
    match = (lo <= target) & (target < hi)  # (..., H, L-1)
    n_match = match.sum(dim=-1)
    # A physically meaningful column is strictly increasing in geopotential; reject the
    # whole column otherwise, even where a unique bracketing layer happens to exist.
    monotone = (phi[..., 1:] > phi[..., :-1]).all(dim=-1, keepdim=True)
    valid = (n_match == 1) & monotone
    layer = torch.argmax(match.to(torch.int8), dim=-1)  # first match; 0 when none
    phi_lo = torch.gather(phi, -1, layer)
    phi_hi = torch.gather(phi, -1, layer + 1)
    denom = torch.where(valid, phi_hi - phi_lo, torch.ones_like(phi_lo))
    weight = torch.where(valid, (phi_target - phi_lo) / denom, torch.zeros_like(phi_lo))
    weight = weight.clamp(0.0, 1.0)
    return layer, weight, valid


def refractivity_at_heights(
    phi: torch.Tensor,
    t: torch.Tensor,
    q: torch.Tensor | None,
    ln_p: torch.Tensor,
    phi_target: torch.Tensor,
    *,
    moist: bool = True,
    interp: InterpMethod = "hydrostatic_shape",
    q_interp: HumidityInterp = "log",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Model-implied refractivity at fixed geopotentials.

    Parameters
    ----------
    phi, t, q : torch.Tensor
        Column values on the pressure ladder, shape ``(..., L)``; ``q`` may be
        ``None`` when ``moist`` is False. Physical units (m^2 s^-2, K, kg/kg).
    ln_p : torch.Tensor
        Natural log of the ladder pressures in hPa, shape ``(L,)``.
    phi_target : torch.Tensor
        Target geopotentials (m^2 s^-2), shape ``(H,)``.
    moist : bool
        Include the water-vapour term.
    interp : {"linear_lnp", "hydrostatic_shape"}
        ``linear_lnp`` interpolates ``ln p`` linearly in geopotential (exact for an
        isothermal layer). ``hydrostatic_shape`` uses the closed-form shape of a layer
        whose virtual temperature varies linearly with geopotential, still matching
        both layer endpoints.
    q_interp : {"linear", "log"}
        Interpolate specific humidity linearly or log-linearly in geopotential. Humidity
        decays roughly exponentially with height, so ``log`` avoids the positive bias a
        linear interpolant has across thick moist layers.

    Returns
    -------
    n_model : torch.Tensor
        Refractivity ``(..., H)`` in N-units (1 where invalid, so ``log`` is finite).
    valid : torch.Tensor
        Bool ``(..., H)`` bracket validity mask.
    """
    layer, w, valid = bracket_column(phi, phi_target)
    t_lo = torch.gather(t, -1, layer)
    t_hi = torch.gather(t, -1, layer + 1)
    t_h = (1.0 - w) * t_lo + w * t_hi
    lnp_lo = ln_p[layer]
    lnp_hi = ln_p[layer + 1]

    if moist:
        if q is None:
            msg = "moist refractivity requires a specific-humidity column"
            raise ValueError(msg)
        q = q.clamp_min(0.0)
        q_lo = torch.gather(q, -1, layer)
        q_hi = torch.gather(q, -1, layer + 1)
        if q_interp == "linear":
            q_h = (1.0 - w) * q_lo + w * q_hi
        elif q_interp == "log":
            ln_q_lo = torch.log(q_lo.clamp_min(Q_FLOOR))
            ln_q_hi = torch.log(q_hi.clamp_min(Q_FLOOR))
            q_h = torch.exp((1.0 - w) * ln_q_lo + w * ln_q_hi)
        else:
            msg = f"unknown q_interp method {q_interp!r}"
            raise ValueError(msg)
    else:
        q_lo = q_hi = q_h = None

    if interp == "linear_lnp":
        frac = w
    elif interp == "hydrostatic_shape":
        # T_v linear in Phi within the layer => ln p(Phi) - ln p_lo proportional to
        # ln(T_v(Phi)/T_v,lo); normalise so the layer endpoints are matched exactly.
        if moist:
            tv_lo = t_lo * (1.0 + 0.608 * q_lo)
            tv_hi = t_hi * (1.0 + 0.608 * q_hi)
        else:
            tv_lo, tv_hi = t_lo, t_hi
        d = (tv_hi - tv_lo) / tv_lo
        small = d.abs() < 1e-6
        d_safe = torch.where(small, torch.ones_like(d), d)
        frac = torch.log1p(w * d_safe) / torch.log1p(d_safe)
        frac = torch.where(small, w, frac)
    else:
        msg = f"unknown interp method {interp!r}"
        raise ValueError(msg)

    ln_p_h = (1.0 - frac) * lnp_lo + frac * lnp_hi
    p_h = torch.exp(ln_p_h)
    n_model = K1_DRY * p_h / t_h
    if moist:
        e_h = q_h * p_h / (EPS_RD_RV + (1.0 - EPS_RD_RV) * q_h)
        n_model = n_model + K2_WET * e_h / (t_h * t_h)
    n_model = torch.where(valid, n_model, torch.ones_like(n_model))
    return n_model, valid


def _to_index_space(layout: IndexSpace | str | None) -> IndexSpace | None:
    if layout is None:
        return None
    return layout if isinstance(layout, IndexSpace) else IndexSpace(str(layout))


class RefractivityOperatorLoss(BaseLoss):
    """Observation-operator loss against GNSS-RO refractivity at fixed geopotential heights.

    The loss reads the predicted ``z_<p>``/``t_<p>``/``q_<p>`` ladder (model-output
    layout, normalised), denormalises it with the dataset normaliser, evaluates the
    Smith-Weintraub refractivity at each configured height and penalises the log
    residual against the ``refrac_<h>`` observation columns of the target (which must
    be declared as ``target`` variables and kept in physical units, normaliser
    method ``none``). Observations are NaN where absent; a height that is not
    bracketed by a monotone model column at a node is masked as well.

    Per level ``l`` with fractional error scale ``sigma_l`` the contribution is the
    node-weighted mean over valid observations of ``huber((r - bias_l) / sigma_l)``,
    and the loss is ``penalty_weight`` times the mean over active levels.

    Example config (inside a ``CombinedLoss``):

    .. code-block:: yaml

        - _target_: anemoi.training.losses.RefractivityOperatorLoss
          scalers: [node_weights]
          moist: false
          levels:
            - {name: refrac_10400, height: 10400, sigma: 0.0051}
            - {name: refrac_13000, height: 13000, sigma: 0.0058}
    """

    name: str = "refractivity"
    factory_context_keys = frozenset({LossFactoryContextKey.DATA_INDICES, LossFactoryContextKey.NORMALIZER})

    def __init__(
        self,
        *,
        levels: list[dict],
        data_indices: IndexCollection,
        normalizer: object | None = None,
        pressure_levels: list[int] | None = None,
        moist: bool = True,
        interp: InterpMethod = "hydrostatic_shape",
        q_interp: HumidityInterp = "log",
        penalty_weight: float = 1.0,
        huber_delta_sigmas: float = 3.0,
        dry_min_height: float = 10400.0,
        geopotential_prefix: str = "z",
        temperature_prefix: str = "t",
        humidity_prefix: str = "q",
        ignore_nans: bool = True,
    ) -> None:
        """Initialise the refractivity operator loss.

        Parameters
        ----------
        levels : list[dict]
            One entry per observation level: ``name`` (target variable, e.g.
            ``refrac_13000``), ``height`` (gpm), ``sigma`` (fractional error scale),
            optional ``active`` (default True) and ``bias`` (fractional offset
            subtracted from the log residual, default 0).
        data_indices : IndexCollection
            Dataset index collection (injected by the loss factory).
        normalizer : object, optional
            Dataset input normaliser exposing ``_norm_mul``/``_norm_add`` over the
            DATA_FULL variable axis (injected by the loss factory).
        pressure_levels : list[int], optional
            Ladder pressures in hPa, surface first; default 1000 ... 50 (13 levels).
        moist : bool
            Include the water-vapour term (requires ``q_<p>`` in the model output).
        interp : {"linear_lnp", "hydrostatic_shape"}
            Pressure interpolation within a layer, see :func:`refractivity_at_heights`.
        q_interp : {"linear", "log"}
            Humidity interpolation within a layer.
        penalty_weight : float
            Multiplier on the final loss.
        huber_delta_sigmas : float
            Huber transition point in units of ``sigma``; ``<= 0`` disables (pure quadratic).
        dry_min_height : float
            With ``moist=False`` every active level must sit at or above this height (gpm).
        geopotential_prefix, temperature_prefix, humidity_prefix : str
            Variable-name prefixes of the ladder.
        ignore_nans : bool
            Kept for interface compatibility; NaN observations are always masked.
        """
        super().__init__(ignore_nans=ignore_nans)
        # Per-observation normalisation needs full-grid counts; force the gather path.
        self.supports_sharding = False

        if normalizer is None or not (hasattr(normalizer, "_norm_mul") and hasattr(normalizer, "_norm_add")):
            msg = (
                "RefractivityOperatorLoss needs the dataset normaliser (exposing _norm_mul/_norm_add); "
                "it is injected by get_loss_function via LossFactoryContextKey.NORMALIZER."
            )
            raise ValueError(msg)
        # Plain reference (not a submodule): the normaliser already lives in the model.
        object.__setattr__(self, "_normalizer", normalizer)

        self.moist = moist
        self.interp: InterpMethod = interp
        self.q_interp: HumidityInterp = q_interp
        self.penalty_weight = float(penalty_weight)
        self.huber_delta_sigmas = float(huber_delta_sigmas)
        pressure_levels = list(pressure_levels or [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50])
        if len(pressure_levels) < 2 or any(a <= b for a, b in pairwise(pressure_levels)):
            msg = f"pressure_levels must be strictly decreasing with at least two entries, got {pressure_levels}"
            raise ValueError(msg)
        self.pressure_levels = pressure_levels

        active = self._validate_levels(levels, moist=moist, dry_min_height=dry_min_height)
        self.observation_variables = [level["name"] for level in active]
        self.heights_gpm = [float(level["height"]) for level in active]
        self._register_state_indices(
            data_indices,
            prefixes=(geopotential_prefix, temperature_prefix, humidity_prefix),
        )
        self._register_observation_indices(data_indices, normalizer)

        self.register_buffer(
            "ln_p",
            torch.log(torch.tensor(pressure_levels, dtype=torch.float64)).to(torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "phi_target",
            torch.tensor([h * G0 for h in self.heights_gpm], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "sigma",
            torch.tensor([float(level["sigma"]) for level in active], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "bias",
            torch.tensor([float(level.get("bias", 0.0)) for level in active], dtype=torch.float32),
            persistent=False,
        )

        # Diagnostics refreshed on every forward (detached).
        self.last_level_losses: torch.Tensor | None = None
        self.last_level_counts: torch.Tensor | None = None
        self.last_unbracketed_fraction: torch.Tensor | None = None

        LOGGER.info(
            "RefractivityOperatorLoss: %d active levels %s, moist=%s, interp=%s, q_interp=%s",
            len(active),
            self.observation_variables,
            moist,
            interp,
            q_interp,
        )

    # ------------------------------------------------------------------ construction helpers
    @staticmethod
    def _validate_levels(levels: list[dict], *, moist: bool, dry_min_height: float) -> list[dict]:
        active = [dict(level) for level in levels if level.get("active", True)]
        if not active:
            msg = "RefractivityOperatorLoss needs at least one active level"
            raise ValueError(msg)
        for level in active:
            missing = [key for key in ("name", "height", "sigma") if key not in level]
            if missing:
                msg = f"level entry {level} is missing {missing}"
                raise ValueError(msg)
            if float(level["sigma"]) <= 0:
                msg = f"sigma must be positive, got {level}"
                raise ValueError(msg)
            if not moist and float(level["height"]) < dry_min_height:
                msg = (
                    f"level {level['name']} at {level['height']} gpm is below dry_min_height={dry_min_height}; "
                    "the dry operator is biased by water vapour there. Use moist=True or deactivate the level."
                )
                raise ValueError(msg)
        return active

    def _register_state_indices(self, data_indices: IndexCollection, *, prefixes: tuple[str, str, str]) -> None:
        """Resolve the z/t(/q) ladder in MODEL_OUTPUT positions and DATA_FULL indices (for denormalisation)."""
        model_pos = data_indices.model.output.name_to_position
        full_idx = data_indices.name_to_index
        attrs = ("z", "t", "q") if self.moist else ("z", "t")
        names = {
            attr: [f"{prefix}_{lvl}" for lvl in self.pressure_levels]
            for attr, prefix in zip(attrs, prefixes, strict=False)
        }
        missing = [n for group in names.values() for n in group if n not in model_pos]
        if missing:
            msg = f"RefractivityOperatorLoss: ladder variables missing from the model output: {missing}"
            raise ValueError(msg)
        self.state_variables = [n for group in names.values() for n in group]
        for attr, group in names.items():
            self.register_buffer(
                f"{attr}_pos",
                torch.tensor([model_pos[n] for n in group], dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                f"{attr}_full_idx",
                torch.tensor([int(full_idx[n]) for n in group], dtype=torch.long),
                persistent=False,
            )

    def _register_observation_indices(self, data_indices: IndexCollection, normalizer: object) -> None:
        """Resolve observation columns per target layout and check they are kept in physical units."""
        out_pos = data_indices.data.output.name_to_position
        full_pos = data_indices.data_full_name_to_position
        missing = [n for n in self.observation_variables if n not in out_pos]
        if missing:
            msg = (
                f"RefractivityOperatorLoss: observation variables {missing} are not in data.output; "
                "declare them under data.<dataset>.target."
            )
            raise ValueError(msg)
        self.register_buffer(
            "obs_pos_data_output",
            torch.tensor([out_pos[n] for n in self.observation_variables], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "obs_pos_data_full",
            torch.tensor([full_pos[n] for n in self.observation_variables], dtype=torch.long),
            persistent=False,
        )
        self._layout_sizes = {
            IndexSpace.DATA_OUTPUT: len(data_indices.data.output.full),
            IndexSpace.DATA_FULL: len(data_indices.name_to_index),
        }
        obs_full_idx = torch.tensor(
            [int(data_indices.name_to_index[n]) for n in self.observation_variables],
            dtype=torch.long,
        )
        mul = normalizer._norm_mul.detach().cpu()[obs_full_idx]
        add = normalizer._norm_add.detach().cpu()[obs_full_idx]
        if not (torch.allclose(mul, torch.ones_like(mul)) and torch.allclose(add, torch.zeros_like(add))):
            msg = (
                "RefractivityOperatorLoss expects the observation columns in physical units: "
                f"set normaliser method 'none' for {self.observation_variables}."
            )
            raise ValueError(msg)

    # ------------------------------------------------------------------ helpers
    def _denormalise(self, pred: torch.Tensor, pos: torch.Tensor, full_idx: torch.Tensor) -> torch.Tensor:
        mul = self._normalizer._norm_mul.to(device=pred.device)[full_idx]
        add = self._normalizer._norm_add.to(device=pred.device)[full_idx]
        x = pred.index_select(-1, pos).to(torch.float32)
        return (x - add.to(torch.float32)) / mul.to(torch.float32)

    def _observation_positions(self, target: torch.Tensor, target_layout: IndexSpace | str | None) -> torch.Tensor:
        layout = _to_index_space(target_layout)
        if layout is None:
            width = target.shape[-1]
            matches = [space for space, size in self._layout_sizes.items() if size == width]
            if len(matches) != 1:
                msg = (
                    f"RefractivityOperatorLoss: cannot infer the target layout from width {width}; "
                    f"known widths {self._layout_sizes}. Pass target_layout explicitly."
                )
                raise ValueError(msg)
            layout = matches[0]
        if layout == IndexSpace.DATA_OUTPUT:
            return self.obs_pos_data_output
        if layout == IndexSpace.DATA_FULL:
            return self.obs_pos_data_full
        msg = f"RefractivityOperatorLoss: target layout {layout.value!r} does not carry observation columns"
        raise ValueError(msg)

    def _huber(self, u: torch.Tensor) -> torch.Tensor:
        k = self.huber_delta_sigmas
        if k <= 0:
            return u * u
        au = u.abs()
        return torch.where(au <= k, u * u, 2.0 * k * au - k * k)

    def model_refractivity(
        self,
        pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return model refractivity ``(..., H)`` and bracket validity for a MODEL_OUTPUT prediction."""
        phi = self._denormalise(pred, self.z_pos, self.z_full_idx)
        t = self._denormalise(pred, self.t_pos, self.t_full_idx)
        q = self._denormalise(pred, self.q_pos, self.q_full_idx) if self.moist else None
        return refractivity_at_heights(
            phi,
            t,
            q,
            self.ln_p,
            self.phi_target,
            moist=self.moist,
            interp=self.interp,
            q_interp=self.q_interp,
        )

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,  # noqa: ARG002
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,  # noqa: ARG002
        squash_mode: Squash_mode = "avg",  # noqa: ARG002
        pred_layout: IndexSpace | str | None = None,
        target_layout: IndexSpace | str | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Compute the refractivity operator loss.

        Parameters
        ----------
        pred : torch.Tensor
            Normalised prediction ``(bs, time, ens, grid, n_model_outputs)``.
        target : torch.Tensor
            Normalised target in DATA_OUTPUT or DATA_FULL layout (observation columns physical).
        squash : bool
            If False, return a MODEL_OUTPUT-width vector with each level's contribution
            deposited on the ladder variables of its typical bracketing layer.
        scaler_indices, without_scalers, grid_shard_slice, group, squash_mode
            Standard loss kwargs; only ``without_scalers``/``grid_shard_slice`` are used
            (node weights). The loss does not support sharded grids.
        pred_layout, target_layout : IndexSpace | str | None
            Variable layouts of ``pred`` (must be MODEL_OUTPUT) and ``target``.

        Returns
        -------
        torch.Tensor
            Scalar loss, or per-model-output vector when ``squash`` is False.
        """
        pl = _to_index_space(pred_layout)
        if pl is not None and pl != IndexSpace.MODEL_OUTPUT:
            msg = f"RefractivityOperatorLoss expects pred_layout=MODEL_OUTPUT, got {pl.value!r}"
            raise ValueError(msg)
        if grid_shard_slice is not None:
            msg = "RefractivityOperatorLoss does not support sharded grids (supports_sharding=False)"
            raise ValueError(msg)

        obs_pos = self._observation_positions(target, target_layout)
        with torch.autocast(device_type=pred.device.type, enabled=False):
            n_obs = target.index_select(-1, obs_pos).to(torch.float32)  # (..., G, H)
            has_obs = torch.isfinite(n_obs) & (n_obs > 0)
            n_model, valid = self.model_refractivity(pred)
            mask = has_obs & valid

            safe_obs = torch.where(has_obs, n_obs, torch.ones_like(n_obs))
            r = torch.log(n_model) - torch.log(safe_obs) - self.bias
            u = self._huber(r / self.sigma)
            u = torch.where(mask, u, torch.zeros_like(u))

            # Node weights (GRID scaler) broadcast over the level axis.
            weights = self.scale(
                torch.ones((*u.shape[:-1], 1), dtype=u.dtype, device=u.device),
                without_scalers=without_scalers,
                grid_shard_slice=None,
            )
            w_mask = weights * mask.to(u.dtype)
            reduce_dims = tuple(range(u.dim() - 1))
            num = (w_mask * u).sum(dim=reduce_dims)  # (H,)
            den = w_mask.sum(dim=reduce_dims)
            level_loss = torch.where(den > 0, num / den.clamp_min(torch.finfo(u.dtype).tiny), torch.zeros_like(num))
            n_active = (den > 0).sum().clamp_min(1)
            loss = self.penalty_weight * level_loss.sum() / n_active

            self.last_level_losses = level_loss.detach()
            self.last_level_counts = mask.sum(dim=reduce_dims).detach()
            n_has = has_obs.sum()
            self.last_unbracketed_fraction = ((has_obs & ~valid).sum() / n_has.clamp_min(1)).detach()

        if squash:
            return loss.to(pred.dtype)

        # Per-variable view: deposit each level on the ladder variables of the layer that
        # brackets the level at the batch-mean column (diagnostic only).
        out = torch.zeros(pred.shape[-1], dtype=pred.dtype, device=pred.device)
        with torch.no_grad():
            phi_mean = self._denormalise(pred, self.z_pos, self.z_full_idx)
            phi_mean = phi_mean.reshape(-1, phi_mean.shape[-1]).mean(dim=0, keepdim=True)
            layer, _, _ = bracket_column(phi_mean, self.phi_target)
            layer = layer.squeeze(0)
        per_level = (self.penalty_weight * level_loss / n_active).to(pred.dtype)
        groups = [self.z_pos, self.t_pos] + ([self.q_pos] if self.moist else [])
        for positions in groups:
            out = out.index_add(0, positions[layer], per_level / len(groups))
            out = out.index_add(0, positions[layer + 1], torch.zeros_like(per_level))
        return out
