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
from typing import TYPE_CHECKING

import torch

from anemoi.training.losses.base import BaseLossWrapper
from anemoi.training.utils.enums import TensorDim

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

    from anemoi.models.data.sources import Source
    from anemoi.training.losses.base import BaseLoss

LOGGER = logging.getLogger(__name__)


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
        pred: Source,
        target: Source,
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
        pred : Source
            Gridded prediction source, data of shape ``(bs, time, ens, latlon, nvar)``.
        target : Source
            Gridded target source, data of shape ``(bs, time, ens, latlon, nvar)``.
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
        if pred.layout.time_in_grid:
            msg = "TimeAggregateLossWrapper needs an explicit time axis; it does not support tabular sources."
            raise NotImplementedError(msg)
        assert (
            pred.time_size > 1
        ), "TimeAggregateLossWrapper requires an output time dimension of size > 1 for aggregation."
        loss = torch.tensor(0.0, dtype=pred.dtype, device=pred.device, requires_grad=False)

        # Exclude the TIME scaler from inner loss calls since we iterate per-step
        # and apply time weights manually.
        without_time = list(without_scalers or [])
        if TensorDim.TIME not in without_time:
            without_time.append(TensorDim.TIME)

        # Extract time weights from the shared scaler (if present)
        time_weights = None
        for dims, scaler in self.loss.scaler.tensors.values():
            if isinstance(dims, int):
                dims = (dims,)
            if TensorDim.TIME.value in dims or TensorDim.TIME in dims:
                time_weights = scaler
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
            loss = loss + self._compute_agg_loss(agg_op, pred, target, time_weights, shared_kwargs)

        # Average over the number of aggregation types, matching the old per-term
        # normalisation (old code: loss /= num_interp_steps + num_aggregate_ops).
        if self.time_aggregation_types:
            loss = loss / len(self.time_aggregation_types)
        return loss

    @staticmethod
    def _time_axis(source: Source) -> int:
        return source.layout.axis(TensorDim.TIME, ndim=source.data.ndim)

    def _compute_agg_loss(
        self,
        agg_op: str,
        pred: Source,
        target: Source,
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
        pred_agg = pred.clone(data=fn(pred.data, dim=self._time_axis(pred), keepdim=True))
        target_agg = target.clone(data=fn(target.data, dim=self._time_axis(target), keepdim=True))
        return self.loss(pred_agg, target_agg, **shared_kwargs)

    def _compute_diff_loss(
        self,
        pred: Source,
        target: Source,
        time_weights: torch.Tensor | None,
        shared_kwargs: dict,
    ) -> torch.Tensor:
        """Compute per-step diff loss, optionally weighted by time scaler."""

        def time_diff(source: Source) -> torch.Tensor:
            axis, n_time = self._time_axis(source), source.time_size
            return source.data.narrow(axis, 1, n_time - 1) - source.data.narrow(axis, 0, n_time - 1)

        pred_diff = time_diff(pred)  # (bs, time-1, ens, latlon, nvar)
        target_diff = time_diff(target)
        pred_axis, target_axis = self._time_axis(pred), self._time_axis(target)
        loss = torch.tensor(0.0, dtype=pred.dtype, device=pred.device, requires_grad=False)
        for step in range(pred_diff.shape[pred_axis]):
            step_loss = self.loss(
                pred.clone(data=pred_diff.narrow(pred_axis, step, 1)),
                target.clone(data=target_diff.narrow(target_axis, step, 1)),
                **shared_kwargs,
            )
            if time_weights is not None:
                step_loss = step_loss * time_weights[step]
            loss = loss + step_loss
        return loss
