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

from anemoi.training.losses.base import FunctionalLoss
from anemoi.training.losses.base import Squash_mode
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


class NaNAwareMSELoss(FunctionalLoss):
    """MSE loss with density-weighted NaN handling.

    This loss compensates for variables with different NaN densities,
    ensuring that sparse variables (with more NaNs) have equal impact
    in the total loss as dense variables.

    For example, if a variable has 50% NaN values, each valid point
    is weighted 2x so that the variable's total contribution matches
    what a fully-dense variable would contribute.

    This is useful when training on datasets where some variables
    have irregular coverage (e.g., satellite observations) alongside
    variables with full coverage (e.g., reanalysis fields).
    """

    name: str = "nan_aware_mse"

    def __init__(self, ignore_nans: bool = False) -> None:
        super().__init__(ignore_nans=ignore_nans)
        # Density weights are computed from per-variable NaN counts along the
        # full grid dimension; per-shard counts would be wrong, so force the
        # gather-before-loss path.
        self.supports_sharding = False

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the squared error.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, output_times, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, output_times, ensemble, lat*lon, n_outputs)

        Returns
        -------
        torch.Tensor
            Squared error
        """
        return torch.square(pred - target)

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
        squash_mode: Squash_mode = "avg",
        **_kwargs,
    ) -> torch.Tensor:
        """Calculates the density-weighted MSE loss.

        NaN values in the target are masked out, and the loss is scaled
        per-variable so that variables with more NaNs still contribute
        equally to the total loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, output_times, ensemble, lat*lon, n_outputs).
        target : torch.Tensor
            Target tensor, shape (bs, output_times, ensemble, lat*lon, n_outputs).
        squash : bool, optional
            Average last dimension, by default True.
        scaler_indices: tuple[int,...], optional
            Indices to subset the calculated scaler with, by default None.
        without_scalers: list[str] | list[int] | None, optional
            list of scalers to exclude from scaling. Can be list of names or dimensions to exclude.
            By default None.
        grid_shard_slice : slice, optional
            Slice of the grid if x comes sharded, by default None.
        group: ProcessGroup, optional
            Distributed group, by default None.
        squash_mode : {"avg", "sum"}, optional
            Reduction mode for the variable dimension, by default ``"avg"``.
        **_kwargs
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            Weighted loss.
        """
        is_sharded = grid_shard_slice is not None

        # Identify NaN positions in target. Density weights are target-driven by
        # design, so only target NaNs are considered here (unlike mask_nans).
        nan_mask = torch.isnan(target)
        latlon = target.shape[TensorDim.GRID]

        # Count NaNs per variable (along the grid dimension)
        nan_per_var = nan_mask.sum(dim=TensorDim.GRID, keepdim=True)

        # Compute density weights: compensate for missing values so sparse variables have equal impact
        # If a variable has 50% NaNs, weight = 2.0; if 0% NaNs, weight = 1.0
        # Clamp denominator to avoid division by zero if a variable is entirely NaN
        valid_points = (latlon - nan_per_var).clamp(min=1)
        density_weights = latlon / valid_points

        # Mask out NaN positions by setting both pred and target to 0 there
        target = target.masked_fill(nan_mask, 0.0)
        pred = pred.masked_fill(nan_mask, 0.0)

        out = self.calculate_difference(pred, target)

        # Apply density weights to compensate for sparse variables
        out = out * density_weights

        out = self.scale(out, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)
        return self.reduce(out, squash, group=group if is_sharded else None, squash_mode=squash_mode)
