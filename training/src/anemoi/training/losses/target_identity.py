# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Identity observation loss: a target variable that lands directly on a model variable."""

from __future__ import annotations

import logging
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

Reduction = Literal["sonde", "per_obs"]


def _to_index_space(layout: IndexSpace | str | None) -> IndexSpace | None:
    if layout is None:
        return None
    return layout if isinstance(layout, IndexSpace) else IndexSpace(str(layout))


class TargetIdentityLoss(BaseLoss):
    """Masked squared (or Huber) error between ``target`` variables and the model variables they observe.

    For each configured pair the physical target column (declared under ``data.<dataset>.target``,
    normaliser ``none``) is normalised with the **model variable's** affine normalisation, so the
    residual is expressed in the same normalised units as the main MSE term on that variable and
    the pair ``weight`` is directly comparable to ``general_variable`` weights. Optionally
    ``sigma`` (physical units) turns the residual into observation-error units instead.

    Use cases: GNSS-RO dry retrievals of ``t``/``z`` at stratospheric pressure levels
    (``gpt_100 -> t_100``, sparse, NaN where absent) or reanalysis fields as dense targets
    (``era_z_500 -> z_500``) to stabilise rollouts.

    ``reduction: sonde`` reproduces the zero-fill semantics of the sonde MSE (masked points
    contribute zero but stay in the grid sum, so sparse sources are implicitly down-weighted by
    coverage, consistently with the existing observation terms); ``per_obs`` divides by the
    valid count instead, as the refractivity operator does.

    Example config (inside a ``CombinedLoss``):

    .. code-block:: yaml

        - _target_: anemoi.training.losses.TargetIdentityLoss
          scalers: [node_weights]
          reduction: sonde
          pairs:
            - {target: era_z_500, model: z_500, weight: 800}
            - {target: era_t_500, model: t_500, weight: 40}
    """

    name: str = "target_identity"
    factory_context_keys = frozenset({LossFactoryContextKey.DATA_INDICES, LossFactoryContextKey.NORMALIZER})

    def __init__(
        self,
        *,
        pairs: list[dict],
        data_indices: IndexCollection,
        normalizer: object | None = None,
        reduction: Reduction = "sonde",
        huber_delta: float | None = None,
        penalty_weight: float = 1.0,
        ignore_nans: bool = True,
    ) -> None:
        """Initialise the identity loss.

        Parameters
        ----------
        pairs : list[dict]
            One entry per observed variable: ``target`` (target variable name), ``model``
            (model-output variable name), optional ``weight`` (default 1) and ``sigma``
            (physical observation error; when given the residual is divided by it instead of
            being expressed in normalised units).
        data_indices : IndexCollection
            Dataset index collection (injected by the loss factory).
        normalizer : object, optional
            Dataset input normaliser exposing ``_norm_mul``/``_norm_add`` (injected).
        reduction : {"sonde", "per_obs"}
            Grid reduction, see class docstring.
        huber_delta : float, optional
            Huber transition (in the residual's units); ``None`` for a pure quadratic.
        penalty_weight : float
            Multiplier on the final loss.
        ignore_nans : bool
            Kept for interface compatibility; NaN targets are always masked.
        """
        super().__init__(ignore_nans=ignore_nans)
        self.supports_sharding = reduction == "sonde"  # per_obs needs full-grid counts
        if normalizer is None or not (hasattr(normalizer, "_norm_mul") and hasattr(normalizer, "_norm_add")):
            msg = (
                "TargetIdentityLoss needs the dataset normaliser (exposing _norm_mul/_norm_add); "
                "it is injected by get_loss_function via LossFactoryContextKey.NORMALIZER."
            )
            raise ValueError(msg)
        object.__setattr__(self, "_normalizer", normalizer)
        if reduction not in ("sonde", "per_obs"):
            msg = f"reduction must be 'sonde' or 'per_obs', got {reduction!r}"
            raise ValueError(msg)
        if not pairs:
            msg = "TargetIdentityLoss needs at least one pair"
            raise ValueError(msg)
        self.reduction: Reduction = reduction
        self.huber_delta = None if huber_delta is None else float(huber_delta)
        self.penalty_weight = float(penalty_weight)

        model_pos = data_indices.model.output.name_to_position
        out_pos = data_indices.data.output.name_to_position
        full_pos = data_indices.data_full_name_to_position
        full_idx = data_indices.name_to_index
        self.observation_variables = [str(p["target"]) for p in pairs]
        self.state_variables = [str(p["model"]) for p in pairs]
        missing = [n for n in self.state_variables if n not in model_pos]
        if missing:
            msg = f"TargetIdentityLoss: model variables missing from the model output: {missing}"
            raise ValueError(msg)
        missing = [n for n in self.observation_variables if n not in out_pos]
        if missing:
            msg = (
                f"TargetIdentityLoss: target variables {missing} are not in data.output; "
                "declare them under data.<dataset>.target."
            )
            raise ValueError(msg)
        self.register_buffer(
            "model_pos",
            torch.tensor([model_pos[n] for n in self.state_variables], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "model_full_idx",
            torch.tensor([int(full_idx[n]) for n in self.state_variables], dtype=torch.long),
            persistent=False,
        )
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
        obs_full_idx = torch.tensor([int(full_idx[n]) for n in self.observation_variables], dtype=torch.long)
        mul = normalizer._norm_mul.detach().cpu()[obs_full_idx]
        add = normalizer._norm_add.detach().cpu()[obs_full_idx]
        if not (torch.allclose(mul, torch.ones_like(mul)) and torch.allclose(add, torch.zeros_like(add))):
            msg = (
                "TargetIdentityLoss expects the target columns in physical units: "
                f"set normaliser method 'none' for {self.observation_variables}."
            )
            raise ValueError(msg)
        self.register_buffer(
            "pair_weight",
            torch.tensor([float(p.get("weight", 1.0)) for p in pairs], dtype=torch.float32),
            persistent=False,
        )
        sigma = [p.get("sigma") for p in pairs]
        self.use_sigma = any(s is not None for s in sigma)
        if self.use_sigma and any(s is None or float(s) <= 0 for s in sigma):
            msg = "TargetIdentityLoss: when sigma is used it must be given (> 0) for every pair"
            raise ValueError(msg)
        self.register_buffer(
            "sigma",
            torch.tensor([float(s) if s is not None else 1.0 for s in sigma], dtype=torch.float32),
            persistent=False,
        )
        self.last_pair_losses: torch.Tensor | None = None
        self.last_pair_counts: torch.Tensor | None = None
        self.last_pair_bias: torch.Tensor | None = None
        LOGGER.info(
            "TargetIdentityLoss: %s, reduction=%s, sigma=%s",
            dict(zip(self.observation_variables, self.state_variables, strict=False)),
            reduction,
            self.use_sigma,
        )

    # ------------------------------------------------------------------ helpers
    def _observation_positions(self, target: torch.Tensor, target_layout: IndexSpace | str | None) -> torch.Tensor:
        layout = _to_index_space(target_layout)
        if layout is None:
            width = target.shape[-1]
            matches = [space for space, size in self._layout_sizes.items() if size == width]
            if len(matches) != 1:
                msg = (
                    f"TargetIdentityLoss: cannot infer the target layout from width {width}; "
                    f"known {self._layout_sizes}."
                )
                raise ValueError(msg)
            layout = matches[0]
        if layout == IndexSpace.DATA_OUTPUT:
            return self.obs_pos_data_output
        if layout == IndexSpace.DATA_FULL:
            return self.obs_pos_data_full
        msg = f"TargetIdentityLoss: target layout {layout.value!r} does not carry observation columns"
        raise ValueError(msg)

    def _affine(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        mul = self._normalizer._norm_mul.to(device=device)[self.model_full_idx].to(torch.float32)
        add = self._normalizer._norm_add.to(device=device)[self.model_full_idx].to(torch.float32)
        return mul, add

    def residual(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_layout: IndexSpace | str | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the per-pair residual ``(..., P)`` (normalised or sigma units) and the validity mask."""
        obs = target.index_select(-1, self._observation_positions(target, target_layout)).to(torch.float32)
        mask = torch.isfinite(obs)
        obs = torch.where(mask, obs, torch.zeros_like(obs))
        mul, add = self._affine(pred.device)
        pred_n = pred.index_select(-1, self.model_pos).to(torch.float32)
        # Either the physical difference in sigma units, or the difference in the model variable's
        # normalised units (target normalised with the model variable's affine).
        r = ((pred_n - add) / mul - obs) / self.sigma if self.use_sigma else pred_n - (obs * mul + add)
        mask = mask & torch.isfinite(pred_n)
        return torch.where(mask, r, torch.zeros_like(r)), mask

    def _rho(self, r: torch.Tensor) -> torch.Tensor:
        if self.huber_delta is None:
            return r * r
        k = self.huber_delta
        ar = r.abs()
        return torch.where(ar <= k, r * r, 2.0 * k * ar - k * k)

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
        """Compute the identity loss.

        Parameters
        ----------
        pred : torch.Tensor
            Normalised prediction ``(bs, time, ens, grid, n_model_outputs)``.
        target : torch.Tensor
            Normalised target in DATA_OUTPUT or DATA_FULL layout (target columns physical).
        squash : bool
            If False, return a MODEL_OUTPUT-width vector with each pair deposited on its model variable.
        scaler_indices, without_scalers, grid_shard_slice, group, squash_mode
            Standard loss kwargs; node weights are applied through ``without_scalers``/``grid_shard_slice``.
        pred_layout, target_layout : IndexSpace | str | None
            Variable layouts of ``pred`` (must be MODEL_OUTPUT) and ``target``.

        Returns
        -------
        torch.Tensor
            Scalar loss, or per-model-output vector when ``squash`` is False.
        """
        pl = _to_index_space(pred_layout)
        if pl is not None and pl != IndexSpace.MODEL_OUTPUT:
            msg = f"TargetIdentityLoss expects pred_layout=MODEL_OUTPUT, got {pl.value!r}"
            raise ValueError(msg)
        with torch.autocast(device_type=pred.device.type, enabled=False):
            r, mask = self.residual(pred, target, target_layout)
            u = torch.where(mask, self._rho(r), torch.zeros_like(r))  # (..., G, P)
            weights = self.scale(
                torch.ones((*u.shape[:-1], 1), dtype=u.dtype, device=u.device),
                without_scalers=without_scalers,
                grid_shard_slice=grid_shard_slice,
            )
            reduce_dims = tuple(range(u.dim() - 1))
            if self.reduction == "sonde":
                # zero-fill semantics: node-weighted sum over the grid, mean over batch/time/ensemble
                n_lead = 1
                for d in u.shape[:-2]:
                    n_lead *= d
                pair_loss = (weights * u).sum(dim=reduce_dims) / n_lead
            else:
                w_mask = weights * mask.to(u.dtype)
                den = w_mask.sum(dim=reduce_dims)
                pair_loss = torch.where(
                    den > 0,
                    (w_mask * u).sum(dim=reduce_dims) / den.clamp_min(torch.finfo(u.dtype).tiny),
                    torch.zeros_like(den),
                )
            weighted = self.pair_weight * pair_loss
            loss = self.penalty_weight * weighted.sum()

            w_mask = weights * mask.to(u.dtype)
            den = w_mask.sum(dim=reduce_dims)
            self.last_pair_losses = pair_loss.detach()
            self.last_pair_counts = mask.sum(dim=reduce_dims).detach()
            self.last_pair_bias = torch.where(
                den > 0,
                (w_mask * r).sum(dim=reduce_dims) / den.clamp_min(torch.finfo(u.dtype).tiny),
                torch.zeros_like(den),
            ).detach()
        if squash:
            return loss.to(pred.dtype)
        out = torch.zeros(pred.shape[-1], dtype=pred.dtype, device=pred.device)
        return out.index_add(0, self.model_pos, (self.penalty_weight * weighted).to(pred.dtype))
