# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch

from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import BaseLossWrapper

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.distributed.distributed_c10d import ProcessGroup

    from anemoi.models.data import Source
    from anemoi.models.data import TensorLayout
    from anemoi.training.losses.base import Squash_mode

LOGGER = logging.getLogger(__name__)


class TargetWeightedLoss(BaseLossWrapper):
    """Wraps a point-wise loss and down-weights the points of selected variables whose target is zero.

    For every variable listed in ``zero_target_weights`` the point-wise error of the inner loss
    (e.g. the squared error of ``MSELoss``) is multiplied by ``1`` where the target is positive
    and by the configured weight where the target is zero (or negative). The weight is passed to the
    inner loss as its ``weights`` argument, so the weighting happens before the scalers and the
    reduction of the inner loss and composes multiplicatively with all scalers (node weights,
    variable weights, ...). Other variables are unchanged, and with an empty ``zero_target_weights``
    the result equals the inner loss.

    This is useful for sparse, non-negative targets (e.g. a mask that is one in a single grid
    cell around an event and zero elsewhere). For a binary target that is 1 with conditional
    probability ``p``, the plain squared error is minimised by ``p`` itself, and with the zero-target
    weight ``w`` the minimiser of the weighted squared error is ``q = p / (p + w * (1 - p))``, i.e. a
    cost-sensitive score and not a calibrated probability. This holds for binary targets only: for
    general positive targets the weight only changes the relative cost of the zero-target points.

    The inner loss must apply ``weights`` to its point-wise error (``applies_weights`` is True): ``MSELoss``,
    ``MAELoss``, ``HuberLoss``, ``LogCoshLoss`` and the ensemble ``CRPS``. Other losses (spectral losses,
    energy and variogram scores, ``RMSELoss``, other wrappers) are rejected.

    The weight is computed from the target, which has no ensemble members. For ``CRPS`` it multiplies the
    point-wise CRPS of the ensemble, which has a single ensemble entry, so every member of the ensemble
    is weighted alike.

    Variable names are read from the ``variables`` of the prediction source at call time, so they refer to the
    variables kept by the ``LossVariableMapper`` that wraps the loss (``predicted_variables`` of the inner
    loss configuration) and subsetting and reordering of variables is supported. A name that is not among
    them raises a ``ValueError`` when the loss is called.
    """

    def __init__(
        self,
        loss_fn: BaseLoss,
        zero_target_weights: Mapping[str, float] | None = None,
        ignore_nans: bool = False,
    ) -> None:
        """Initialise the wrapper.

        Parameters
        ----------
        loss_fn : BaseLoss
            Inner point-wise loss that applies ``weights`` (``applies_weights`` is True).
        zero_target_weights : Mapping[str, float] | None
            Variable name -> weight applied to the point-wise error where the target is not positive.
            Weights must be finite and strictly positive. Positive targets have weight 1.
            If None or empty, the result equals the inner loss.
        ignore_nans : bool, optional
            Allow NaNs in the loss and ignore them, by default False. Must agree with the inner loss.
            NaN targets are set to zero (together with the prediction) by the inner loss, so they
            contribute no error whatever their weight.
        """
        super().__init__(loss=loss_fn, ignore_nans=ignore_nans)

        if not self._is_pointwise(loss_fn):
            msg = (
                "TargetWeightedLoss needs a point-wise inner loss that applies `weights` "
                f"(e.g. MSELoss, MAELoss, HuberLoss or CRPS), got {type(loss_fn).__name__}."
            )
            raise TypeError(msg)

        zero_target_weights = dict(zero_target_weights or {})
        for variable, weight in zero_target_weights.items():
            if isinstance(weight, bool) or not isinstance(weight, int | float):
                msg = f"zero_target_weights[{variable!r}] must be a number, got {weight!r}."
                raise TypeError(msg)
            if not math.isfinite(weight) or weight <= 0:
                msg = f"zero_target_weights[{variable!r}] must be finite and strictly positive, got {weight!r}."
                raise ValueError(msg)

        self.zero_target_weights: dict[str, float] = {k: float(v) for k, v in zero_target_weights.items()}
        # Zero-target weight per column, for the variables and device seen last
        self._zero_weight_cache: dict[tuple[tuple[str, ...], str], torch.Tensor] = {}

    @staticmethod
    def _is_pointwise(loss: BaseLoss) -> bool:
        """Whether the loss weights its point-wise error before scaling and reduction through ``weights``."""
        return loss.applies_weights

    def _zero_weights(self, variables: list[str], reference: torch.Tensor, layout: TensorLayout) -> torch.Tensor:
        """Zero-target weight of each variable (1 if not configured), shaped to broadcast against ``reference``."""
        names = tuple(variables)
        missing = [name for name in self.zero_target_weights if name not in names]
        if missing:
            msg = (
                f"zero_target_weights variables {missing} are not among the variables of the loss: {list(names)}. "
                "Check `predicted_variables` of the inner loss."
            )
            raise ValueError(msg)

        key = (names, str(reference.device))
        zero_weights = self._zero_weight_cache.get(key)
        if zero_weights is None:
            zero_weights = torch.tensor(
                [self.zero_target_weights.get(name, 1.0) for name in names],
                dtype=torch.float32,
                device=reference.device,
            )
            self._zero_weight_cache[key] = zero_weights

        shape = [1] * reference.ndim
        shape[layout.axis("variables", ndim=reference.ndim)] = -1
        return zero_weights.view(shape)

    def _target_weights(self, variables: list[str], target: torch.Tensor, layout: TensorLayout) -> torch.Tensor:
        """Weight of every target point: 1 where the target is positive, the zero-target weight elsewhere."""
        zero_weights = self._zero_weights(variables, target, layout)
        # NaN targets are not positive: whatever their weight, the inner loss gives them no error
        return torch.where(target > 0, torch.ones_like(zero_weights), zero_weights)

    def forward(
        self,
        pred: Source,
        target: Source,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        squash_mode: Squash_mode | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Calculate the inner loss with the point-wise error weighted by the target.

        Parameters
        ----------
        pred : Source
            Prediction source (gridded or tabular).
        target : Source
            Target source, same layout as ``pred``. Its ``variables`` name the columns.
        squash : bool, optional
            Average the variable dimension, by default True.
        scaler_indices : tuple[int, ...], optional
            Indices to subset the calculated scaler with, by default None.
        without_scalers : list[str] | list[int] | None, optional
            Scalers to exclude from scaling, by default None.
        grid_shard_slice : slice, optional
            Slice of the grid if the tensors come sharded, by default None.
        group : ProcessGroup, optional
            Distributed group to reduce over, by default None.
        squash_mode : {"avg", "sum"}, optional
            Reduction mode for the variable dimension. If omitted, the inner loss default ("avg") is used.
        **kwargs
            Additional keyword arguments for the inner loss. A ``weights`` argument (or per-sample
            ``weights`` in ``per_sample_kwargs`` for tabular sources) is multiplied with the target weights.

        Returns
        -------
        torch.Tensor
            Weighted loss.
        """
        if self.zero_target_weights:
            # Keyed by the predicted variable, as ``zero_target_weights`` are
            variables = list(pred.variables)

            if target.is_tabular:
                # One call per sample, so the weights are per-sample arguments
                per_sample_kwargs = dict(kwargs.pop("per_sample_kwargs", None) or {})
                sample_weights = [self._target_weights(variables, sample, target.layout) for sample in target.values]
                given = per_sample_kwargs.get("weights")
                if given is not None:
                    sample_weights = [own * other for own, other in zip(sample_weights, given, strict=True)]
                per_sample_kwargs["weights"] = sample_weights
                kwargs["per_sample_kwargs"] = per_sample_kwargs
            else:
                target_weights = self._target_weights(variables, target.values, target.layout)
                given = kwargs.get("weights")
                kwargs["weights"] = target_weights if given is None else target_weights * given

        if squash_mode is not None:
            kwargs["squash_mode"] = squash_mode

        return self.loss(
            pred,
            target,
            squash=squash,
            scaler_indices=scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
            group=group,
            **kwargs,
        )
