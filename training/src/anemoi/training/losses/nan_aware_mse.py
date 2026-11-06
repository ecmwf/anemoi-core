# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.data import TensorLayout
from anemoi.training.losses.base import Squash_mode
from anemoi.training.losses.mse import MSELoss

LOGGER = logging.getLogger(__name__)


class NaNAwareMSELoss(MSELoss):
    """MSE loss with density-weighted NaN handling (ported from feat/testeos).

    On a gridded target, missing (NaN) targets are left out and each variable is rescaled by
    ``num_grid_points / num_valid_points``, per sample, output step and member. With unit-sum
    node weights the loss of a variable is then the (node-weighted) mean squared error over its
    observed points, so a variable observed at 1 % of the grid weighs as much as a complete one.
    The plain ``MSELoss`` with ``ignore_nans`` sums over the observed points instead, which scales
    each variable by its coverage.

    Tabular targets are passed to ``MSELoss``, which already averages over the observations that
    exist.
    """

    name: str = "nan_aware_mse"

    def __init__(self, ignore_nans: bool = False) -> None:
        super().__init__(ignore_nans=ignore_nans)
        # The density weights count NaNs over the full grid; per-shard counts would be wrong,
        # so the loss is computed on the gathered grid.
        self.supports_sharding = False

    def _forward_impl(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        layout: TensorLayout,
        squash: bool = True,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        squash_mode: Squash_mode = "avg",
        valid_counts: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Calculate the density-weighted MSE of one gridded or tabular tensor.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor in ``layout``.
        target : torch.Tensor
            Target tensor in ``layout``; NaN marks a missing target.
        layout : TensorLayout
            Layout describing the logical axes of pred / target.
        squash : bool, optional
            Reduce the variable dimension, by default True.
        scaler_indices : tuple[int, ...], optional
            Indices to subset the calculated scaler with, by default None.
        without_scalers : list[str] | list[int] | None, optional
            Scalers to exclude, by name or dimension. By default None.
        grid_shard_slice : slice, optional
            Slice of the grid if the tensors come sharded, by default None.
        group : ProcessGroup, optional
            Distributed group to reduce over, by default None.
        squash_mode : {"avg", "sum"}, optional
            Reduction mode for the variable dimension, by default ``"avg"``.
        valid_counts : torch.Tensor, optional
            Tabular observations only: per-variable observation counts, see :meth:`BaseLoss.reduce`.
        **kwargs
            Passed on to ``MSELoss`` for tabular tensors.

        Returns
        -------
        torch.Tensor
            Weighted loss.
        """
        if not layout.has_axis("time"):
            return super()._forward_impl(
                pred,
                target,
                layout,
                squash=squash,
                scaler_indices=scaler_indices,
                without_scalers=without_scalers,
                grid_shard_slice=grid_shard_slice,
                group=group,
                squash_mode=squash_mode,
                valid_counts=valid_counts,
                **kwargs,
            )

        is_sharded = grid_shard_slice is not None
        grid = layout.grid

        # Density weights are target-driven: only target NaNs count.
        nan_mask = torch.isnan(target)
        num_points = target.shape[grid]
        valid_points = (num_points - nan_mask.sum(dim=grid, keepdim=True)).clamp(min=1)
        density_weights = num_points / valid_points

        target = target.masked_fill(nan_mask, 0.0)
        pred = pred.masked_fill(nan_mask, 0.0)

        out = self.calculate_difference(pred, target) * density_weights
        out = self.scale(
            out,
            scaler_indices,
            layout=layout,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )
        return self.reduce(
            out,
            layout=layout,
            squash=squash,
            group=group if is_sharded else None,
            squash_mode=squash_mode,
        )
