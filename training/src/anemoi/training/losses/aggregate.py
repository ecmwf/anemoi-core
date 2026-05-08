from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from anemoi.training.losses.base import BaseLossWrapper
from anemoi.training.utils.enums import TensorDim

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

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
            without_time = list(without_time) + [TensorDim.TIME.value]

        # Extract time weights from the shared scaler (if present)
        time_weights = None
        for _name, (dims, scaler) in self.loss.scaler.tensors.items():
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
            if agg_op == "diff":
                pred_agg = pred[:, 1:, ...] - pred[:, :-1, ...]  # (bs, time-1, ens, latlon, nvar)
                target_agg = target[:, 1:, ...] - target[:, :-1, ...]  # (bs, time-1, latlon, nvar)
                # Compute loss per diff-step, weighted by time scaler
                for step in range(pred_agg.shape[1]):
                    step_loss = self.loss(
                        pred_agg[:, step : step + 1, ...],
                        target_agg[:, step : step + 1, ...],
                        **shared_kwargs,
                    )
                    if time_weights is not None:
                        step_loss = step_loss * time_weights[step]
                    loss = loss + step_loss
            elif agg_op == "mean":
                pred_agg = torch.mean(pred, dim=1, keepdim=True)  # (bs, 1, ens, latlon, nvar)
                target_agg = torch.mean(target, dim=1, keepdim=True)  # (bs, 1, latlon, nvar)
                loss = loss + self.loss(pred_agg, target_agg, **shared_kwargs)
            elif agg_op == "min":
                pred_agg = torch.amin(pred, dim=1, keepdim=True)  # (bs, 1, ens, latlon, nvar)
                target_agg = torch.amin(target, dim=1, keepdim=True)  # (bs, 1, latlon, nvar)
                loss = loss + self.loss(pred_agg, target_agg, **shared_kwargs)
            elif agg_op == "max":
                pred_agg = torch.amax(pred, dim=1, keepdim=True)  # (bs, 1, ens, latlon, nvar)
                target_agg = torch.amax(target, dim=1, keepdim=True)  # (bs, 1, latlon, nvar)
                loss = loss + self.loss(pred_agg, target_agg, **shared_kwargs)
            else:
                msg = f"Unknown aggregation type '{agg_op}'. Supported: 'diff', 'mean', 'min', 'max'."
                raise ValueError(msg)

        return loss
