from __future__ import annotations

import logging

import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.training.losses.base import BaseLoss


LOGGER = logging.getLogger(__name__)


class AggregateLossWrapper(BaseLoss):
    """Wraps a base loss and applies it to time-aggregated predictions.

    For each aggregation type in ``aggregation_types``, the wrapper
    transforms ``pred`` (shape ``(bs, time, ens, latlon, nvar)``) and
    ``target`` (shape ``(bs, time, latlon, nvar)``) before delegating to
    the inner ``loss_fn``.  Supported aggregation types:

    - ``"diff"``  – temporal differences (``pred[:, 1:] - pred[:, :-1]``)
    - ``"mean"``, ``"min"``, ``"max"`` – reduction over the time dimension
    """

    def __init__(
        self,
        aggregation_types: list[str],
        loss_fn: BaseLoss,
        ignore_nans: bool = False,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans)
        self.aggregation_types = aggregation_types
        self.loss_fn = loss_fn

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
        """Compute the aggregate loss over all aggregation types.

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
        squash_mode : str, optional
            Variable-dimension reduction mode, by default ``"avg"``.

        Returns
        -------
        torch.Tensor
            Accumulated loss across all aggregation types.
        """
        loss = torch.zeros(1, dtype=pred.dtype, device=pred.device, requires_grad=False)

        shared_kwargs = dict(
            squash=squash,
            scaler_indices=scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
            group=group,
            squash_mode=squash_mode,
            **kwargs,
        )

        for agg_op in self.aggregation_types:
            if agg_op == "diff":
                pred_agg = pred[:, 1:, ...] - pred[:, :-1, ...]  # (bs, time-1, ens, latlon, nvar)
                target_agg = target[:, 1:, ...] - target[:, :-1, ...]  # (bs, time-1, latlon, nvar)
            elif agg_op in {"mean", "min", "max"}:
                agg_fn = getattr(torch, agg_op)
                pred_agg = agg_fn(pred, dim=1)  # (bs, ens, latlon, nvar)
                target_agg = agg_fn(target, dim=1)  # (bs, latlon, nvar)
                if agg_op in {"max", "min"}:
                    pred_agg = pred_agg[0]  # discard indices
                    target_agg = target_agg[0]
            else:
                msg = f"Unknown aggregation type '{agg_op}'. Supported: 'diff', 'mean', 'min', 'max'."
                raise ValueError(msg)

            loss = loss + self.loss_fn(pred_agg, target_agg, **shared_kwargs)

        return loss
