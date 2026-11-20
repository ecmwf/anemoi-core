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
from anemoi.training.losses.base import FunctionalLoss

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.distributed.distributed_c10d import ProcessGroup

    from anemoi.training.losses.base import Squash_mode

LOGGER = logging.getLogger(__name__)


class TargetWeightedLoss(BaseLossWrapper):
    """Wraps a point-wise loss and down-weights the points of selected variables whose target is zero.

    For every variable listed in ``zero_target_weights`` the point-wise error of the inner loss
    (e.g. the squared error of ``MSELoss``) is multiplied by ``1`` where the target is positive
    and by the configured weight where the target is zero (or negative). The weighting happens
    before the scalers and the reduction of the inner loss, so it composes multiplicatively with
    all scalers (node weights, variable weights, ...). Other variables are unchanged, and with an
    empty ``zero_target_weights`` the result equals the inner loss.

    This is useful for sparse, non-negative targets (e.g. a mask that is one in a single grid
    cell around an event and zero elsewhere). For a binary target that is 1 with conditional
    probability ``p``, the plain squared error is minimised by ``p`` itself, and with the zero-target
    weight ``w`` the minimiser of the weighted squared error is ``q = p / (p + w * (1 - p))``, i.e. a
    cost-sensitive score and not a calibrated probability. This holds for binary targets only: for
    general positive targets the weight only changes the relative cost of the zero-target points.

    The inner loss must be a ``FunctionalLoss`` that does not override ``forward``: its ``pointwise_loss``
    hook gives a value per sample, time step, grid point and variable, before the scalers and the
    reduction (``MSELoss``, ``MAELoss``, ``HuberLoss``, ``LogCoshLoss`` and the ensemble ``CRPS``).
    Losses that are not point-wise (spectral losses, energy and variogram scores, ``RMSELoss``,
    ``WeightedMSELoss``, other wrappers) are rejected.

    Variable names are resolved to columns through the ``LossVariableMapper`` that wraps the loss
    (which calls ``set_variable_names``) and refer to the variables kept by the mapper
    (``predicted_variables`` of the inner loss configuration), so subsetting and reordering of
    variables is supported.
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
            Inner point-wise loss (``FunctionalLoss`` with the standard ``forward``).
        zero_target_weights : Mapping[str, float] | None
            Variable name -> weight applied to the point-wise error where the target is not positive.
            Weights must be finite and strictly positive. Positive targets have weight 1.
            If None or empty, the result equals the inner loss.
        ignore_nans : bool, optional
            Allow NaNs in the loss and ignore them, by default False. Must agree with the inner loss.
            NaN targets are set to zero (together with the prediction) before the weighting, so they
            contribute no error.
        """
        super().__init__(loss=loss_fn, ignore_nans=ignore_nans)

        if not self._is_pointwise(loss_fn):
            msg = (
                "TargetWeightedLoss needs a point-wise inner loss (a FunctionalLoss with the standard forward, "
                f"e.g. MSELoss, MAELoss, HuberLoss or CRPS), got {type(loss_fn).__name__}."
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
        self._columns: list[int] | None = None
        self._column_weights: list[float] | None = None
        # Device specific copies of the above, built on first use
        self._cached_index: torch.Tensor | None = None
        self._cached_weights: torch.Tensor | None = None

    @staticmethod
    def _is_pointwise(loss: BaseLoss) -> bool:
        """Whether the loss is point-wise before scaling and reduction, in a form this wrapper can weight."""
        return isinstance(loss, FunctionalLoss) and type(loss).forward is FunctionalLoss.forward

    def set_variable_names(self, names: list[str]) -> None:
        """Resolve the configured variables to columns of the tensors given to the loss.

        Called by ``LossVariableMapper`` with the names of the variables it passes on, in order. The names
        are forwarded to the inner loss as well.

        Parameters
        ----------
        names : list[str]
            Names of the columns of the prediction and target tensors, in order.

        Raises
        ------
        ValueError
            If a configured variable is not one of ``names``.
        """
        super().set_variable_names(names)
        names = list(names)
        # No columns until the names are known to contain every configured variable
        self._columns = None
        self._column_weights = None
        self._cached_index = None
        self._cached_weights = None

        missing = [name for name in self.zero_target_weights if name not in names]
        if missing:
            msg = (
                f"zero_target_weights variables {missing} are not among the variables of the loss: {names}. "
                "Check `predicted_variables` of the inner loss."
            )
            raise ValueError(msg)

        self._columns = [names.index(name) for name in self.zero_target_weights]
        self._column_weights = list(self.zero_target_weights.values())

    def _device_tensors(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the column indices and zero weights on the device and dtype of ``reference``."""
        if (
            self._cached_index is None
            or self._cached_weights is None
            or self._cached_index.device != reference.device
            or self._cached_weights.dtype != reference.dtype
        ):
            self._cached_index = torch.as_tensor(self._columns, dtype=torch.long, device=reference.device)
            self._cached_weights = torch.as_tensor(self._column_weights, dtype=reference.dtype, device=reference.device)
        return self._cached_index, self._cached_weights

    def _weight_error(self, error: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Multiply the point-wise error of the configured columns by 1 (target > 0) or the zero weight."""
        if not self.zero_target_weights:
            return error

        if self._columns is None:
            msg = (
                "TargetWeightedLoss does not know the variable names (`set_variable_names` was never called or "
                "failed). Build it with `get_loss_function` and data indices, or wrap it in a LossVariableMapper "
                "with data indices, or call `set_variable_names`."
            )
            raise RuntimeError(msg)

        index, zero_weights = self._device_tensors(error)
        selected_target = target.index_select(-1, index)
        weights = torch.where(selected_target > 0, torch.ones_like(selected_target), zero_weights)
        # Out of place, so autograd and torch.compile see a plain functional update
        return error.index_copy(-1, index, error.index_select(-1, index) * weights)

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
        squash_mode: Squash_mode | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Calculate the inner loss with the point-wise error weighted by the target.

        Follows the forward of the inner loss: mask NaNs, point-wise loss, weighting by the target,
        scaling, reduction.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, time, ensemble, lat*lon, n_outputs).
        target : torch.Tensor
            Target tensor, same layout as ``pred``.
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
        **_kwargs
            Additional keyword arguments, ignored.

        Returns
        -------
        torch.Tensor
            Weighted loss.
        """
        inner = self.loss
        is_sharded = grid_shard_slice is not None

        pred, target = inner.mask_nans(pred, target)

        out = inner.pointwise_loss(pred, target)
        out = self._weight_error(out, target)
        out = inner.scale(out, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)
        return inner.reduce(
            out,
            squash,
            group=group if is_sharded else None,
            squash_mode="avg" if squash_mode is None else squash_mode,
        )
