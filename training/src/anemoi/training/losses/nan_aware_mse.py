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
from torch_geometric.data import HeteroData

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


class BandNaNAwareMSELoss(NaNAwareMSELoss):
    """NaNAwareMSELoss restricted to a latitude band.

    Targets outside ``lat_min <= latitude < lat_max`` are treated as missing, so the per-variable
    valid-count normalisation of :class:`NaNAwareMSELoss` runs over the band only and the result is
    the mean squared error over the band's valid observations (times the node weights). Use it as
    extra ``validation_metrics`` entries to follow NH / Tropics / SH skill separately.

    A sample with no valid observation of a variable in the band contributes 0 for that variable
    (e.g. SH radiosonde variables at 06/18 UTC). The logged epoch mean is then lower than the mean
    over observed samples, but the same samples are empty in every run, so runs remain comparable.
    """

    name: str = "band_nan_aware_mse"
    needs_graph_data = True
    needs_data_node_name = True

    def __init__(
        self,
        lat_min: float,
        lat_max: float,
        graph_data: HeteroData | None = None,
        data_node_name: str | None = None,
        ignore_nans: bool = False,
        metric_keys: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Initialise BandNaNAwareMSELoss.

        Parameters
        ----------
        lat_min, lat_max : float
            Band edges in degrees; ``lat_max >= 90`` includes the pole.
        graph_data : HeteroData
            Graph whose ``data_node_name`` nodes hold [lat, lon] in radians, injected by the loss
            factory.
        data_node_name : str
            Name of the data nodes, injected by the loss factory.
        ignore_nans : bool, optional
            Passed to :class:`NaNAwareMSELoss`, by default False.
        metric_keys : list[str], optional
            As a validation metric, log only these metric keys: variable names listed in
            ``training.metrics`` (e.g. ``z_500``) or group keys (e.g. ``pl_t``, ``all``). Default:
            every metric range, like the other metrics.
        """
        super().__init__(ignore_nans=ignore_nans)
        del kwargs
        if graph_data is None or data_node_name is None:
            msg = f"{self.__class__.__name__} needs graph_data and data_node_name (injected by the loss factory)."
            raise ValueError(msg)
        if lat_min >= lat_max:
            msg = f"{self.__class__.__name__}: lat_min={lat_min} must be below lat_max={lat_max}."
            raise ValueError(msg)
        latitudes = torch.rad2deg(graph_data[data_node_name].x[:, 0].double())
        in_band = (latitudes >= lat_min) & ((latitudes < lat_max) | (lat_max >= 90.0))
        if not in_band.any():
            msg = f"{self.__class__.__name__}: no grid point between {lat_min} and {lat_max} degrees."
            raise ValueError(msg)
        self.register_buffer("in_band", in_band, persistent=False)
        self.metric_keys = list(metric_keys) if metric_keys is not None else None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        grid_shard_slice: slice | None = None,
        **kwargs,
    ) -> torch.Tensor:
        in_band = self.in_band if grid_shard_slice is None else self.in_band[grid_shard_slice]
        shape = [1] * target.ndim
        shape[TensorDim.GRID] = -1
        target = target.masked_fill(~in_band.view(shape), torch.nan)
        return super().forward(pred, target, squash, grid_shard_slice=grid_shard_slice, **kwargs)
