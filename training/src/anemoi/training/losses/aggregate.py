# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING

import torch

from anemoi.training.losses.base import BaseLossWrapper
from anemoi.training.utils.enums import TensorDim

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

    from anemoi.training.losses.base import BaseLoss

LOGGER = logging.getLogger(__name__)


def _is_time_additive(loss: BaseLoss) -> bool:
    """Whether ``loss`` can be computed for all time steps in a single call.

    True when the loss of a multi-step input equals the sum of the per-step losses
    (``BaseLoss.reduce`` sums over time) and is zero where prediction and target are
    both zero. Unknown losses return False and keep the per-step loop.
    """
    # Imported here to avoid circular imports through the loss factory.
    from anemoi.training.losses.combined import CombinedLoss
    from anemoi.training.losses.huber import HuberLoss
    from anemoi.training.losses.kcrps import CRPS
    from anemoi.training.losses.mae import MAELoss
    from anemoi.training.losses.mse import MSELoss
    from anemoi.training.losses.multiscale import MultiscaleLossWrapper
    from anemoi.training.losses.variable_mapper import LossVariableMapper

    # Exact type matches: subclasses may change the reduction (e.g. RMSELoss takes a
    # square root after summing over time, so it is not additive).
    if type(loss) in (MultiscaleLossWrapper, LossVariableMapper):
        return _is_time_additive(loss.loss)
    if type(loss) is CombinedLoss:
        return all(_is_time_additive(sub_loss) for sub_loss in loss.losses)
    return type(loss) in (MSELoss, MAELoss, HuberLoss, CRPS)


class TimeAggregateLossWrapper(BaseLossWrapper):
    """Wraps a base loss and applies it to time-aggregated predictions.

    Supported time aggregation types:

    - ``"diff"``  - temporal differences (``pred[:, 1:] - pred[:, :-1]``)
    - ``"mean"``, ``"min"``, ``"max"`` - applied over the time window
    """

    def __init__(
        self,
        time_aggregation_types: list[str],
        loss_fn: BaseLoss,
        ignore_nans: bool = False,
    ) -> None:
        super().__init__(loss=loss_fn, ignore_nans=ignore_nans)
        self.time_aggregation_types = time_aggregation_types

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
        squash_mode: str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the time aggregate loss over all time aggregation types.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape ``(bs, time, ens, latlon, nvar)``.
        target : torch.Tensor
            Target tensor, shape ``(bs, time, latlon, nvar)``.
        squash : bool, optional
            Average the variable dimension, by default ``True``.
        scaler_indices : tuple[int, ...] | None, optional
            Indices to subset the scaler, by default ``None``.
        without_scalers : list[str] | list[int] | None, optional
            Scalers to exclude, by default ``None``.
        grid_shard_slice : slice | None, optional
            Grid shard slice, by default ``None``.
        group : ProcessGroup | None, optional
            Distributed group for reduction, by default ``None``.
        squash_mode : str | None, optional
            Variable-dimension reduction mode. If omitted, the wrapped loss default is used.

        Returns
        -------
        torch.Tensor
            Accumulated loss across all aggregation types.
        """
        assert (
            pred.shape[1] > 1
        ), "TimeAggregateLossWrapper requires an output time dimension of size > 1 for aggregation."
        loss = torch.tensor(0.0, dtype=pred.dtype, device=pred.device, requires_grad=False)

        # Exclude the TIME scaler from inner loss calls since we iterate per-step
        # and apply time weights manually.
        without_time = without_scalers or []
        if TensorDim.TIME not in without_time and TensorDim.TIME.value not in without_time:
            without_time = [*list(without_time), TensorDim.TIME.value]

        # Extract time weights from the shared scaler (if present)
        time_weights = None
        time_scaler_name = None
        for name, (dims, scaler) in self.loss.scaler.tensors.items():
            if isinstance(dims, int):
                dims = (dims,)
            if TensorDim.TIME.value in dims or TensorDim.TIME in dims:
                time_weights = scaler
                time_scaler_name = name
                break

        shared_kwargs = dict(
            squash=squash,
            scaler_indices=scaler_indices,
            without_scalers=without_time,
            grid_shard_slice=grid_shard_slice,
            group=group,
            **kwargs,
        )
        if squash_mode is not None:
            shared_kwargs["squash_mode"] = squash_mode

        for agg_op in self.time_aggregation_types:
            if agg_op == "diff":
                loss = loss + self._compute_diff_loss(
                    pred,
                    target,
                    time_weights,
                    shared_kwargs,
                    time_scaler_name=time_scaler_name,
                    without_scalers=without_scalers,
                )
            else:
                loss = loss + self._compute_agg_loss(agg_op, pred, target, time_weights, shared_kwargs)

        # Average over the number of aggregation types, matching the old per-term
        # normalisation (old code: loss /= num_interp_steps + num_aggregate_ops).
        if self.time_aggregation_types:
            loss = loss / len(self.time_aggregation_types)
        return loss

    def _compute_agg_loss(
        self,
        agg_op: str,
        pred: torch.Tensor,
        target: torch.Tensor,
        time_weights: torch.Tensor | None,
        shared_kwargs: dict,
    ) -> torch.Tensor:
        """Compute loss for a single aggregation operation."""
        if agg_op == "diff":
            return self._compute_diff_loss(pred, target, time_weights, shared_kwargs)
        agg_fns = {"mean": torch.mean, "min": torch.amin, "max": torch.amax}
        if agg_op not in agg_fns:
            msg = f"Unknown aggregation type '{agg_op}'. Supported: 'diff', 'mean', 'min', 'max'."
            raise ValueError(msg)
        fn = agg_fns[agg_op]
        pred_agg = fn(pred, dim=1, keepdim=True)
        target_agg = fn(target, dim=1, keepdim=True)
        return self.loss(pred_agg, target_agg, **shared_kwargs)

    @cached_property
    def _inner_loss_is_time_additive(self) -> bool:
        return _is_time_additive(self.loss)

    def _use_single_diff_call(
        self,
        time_scaler_name: str | None,
        without_scalers: list[str] | list[int] | None,
        num_times: int,
    ) -> bool:
        """Whether the diff loss of all steps can be computed in one call to the inner loss."""
        if not self._inner_loss_is_time_additive:
            return False
        if time_scaler_name is None:
            return True

        # The time weights must come from a single 1-D TIME scaler covering all output times,
        # so that the inner loss applies exactly the weights the per-step loop would.
        time_scalers = self.loss.scaler.subset_by_dim(TensorDim.TIME.value).tensors
        if len(time_scalers) != 1 or time_scaler_name not in time_scalers:
            return False
        dims, scaler = time_scalers[time_scaler_name]
        if isinstance(dims, int):
            dims = (dims,)
        if tuple(int(d) for d in dims) != (TensorDim.TIME.value,) or tuple(scaler.shape) != (num_times,):
            return False

        # The per-step loop applies the time weights even when the caller excludes the TIME
        # scaler; the single call applies them through the scaler, so keep the loop then.
        if without_scalers:
            return time_scaler_name in self.loss.scaler.without(list(without_scalers))
        return True

    def _compute_diff_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        time_weights: torch.Tensor | None,
        shared_kwargs: dict,
        *,
        time_scaler_name: str | None = None,
        without_scalers: list[str] | list[int] | None = None,
    ) -> torch.Tensor:
        """Compute per-step diff loss, optionally weighted by time scaler."""
        pred_agg = pred[:, 1:, ...] - pred[:, :-1, ...]  # (bs, time-1, ens, latlon, nvar)
        target_agg = target[:, 1:, ...] - target[:, :-1, ...]  # (bs, time-1, latlon, nvar)

        if self._use_single_diff_call(time_scaler_name, without_scalers, pred.shape[1]):
            if time_scaler_name is None:
                # BaseLoss.reduce sums over time, so one call over all steps gives the sum of
                # the per-step losses.
                return self.loss(pred_agg, target_agg, **shared_kwargs)

            # Append a zero step so the time-1 diffs line up with the TIME scaler (length time)
            # and let the inner loss apply the weights: step s is weighted by time_weights[s],
            # as in the loop, and the zero step adds zero loss.
            pred_agg = torch.cat([pred_agg, torch.zeros_like(pred_agg[:, :1])], dim=1)
            target_agg = torch.cat([target_agg, torch.zeros_like(target_agg[:, :1])], dim=1)
            return self.loss(pred_agg, target_agg, **{**shared_kwargs, "without_scalers": without_scalers})

        loss = torch.tensor(0.0, dtype=pred.dtype, device=pred.device, requires_grad=False)
        for step in range(pred_agg.shape[1]):
            step_loss = self.loss(
                pred_agg[:, step : step + 1, ...],
                target_agg[:, step : step + 1, ...],
                **shared_kwargs,
            )
            if time_weights is not None:
                step_loss = step_loss * time_weights[step]
            loss = loss + step_loss
        return loss
