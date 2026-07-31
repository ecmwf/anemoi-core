# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from contextlib import nullcontext
from typing import Literal

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import all_to_all_transpose
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import reduce_tensor
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import Squash_mode
from anemoi.training.losses.scaler_tensor import ScaleTensor
from anemoi.training.utils.enums import TensorDim

EnergyScoreNorm = Literal["spatial", "variables", "spatial_and_variables"]


class EnergyScoreLoss(BaseLoss):
    """Energy score over space, variables, or their joint field.

    ``norm_over="spatial"`` calculates a spatial norm independently for each
    variable. ``norm_over="variables"`` calculates a joint variable norm at
    every grid node. ``norm_over="spatial_and_variables"`` calculates one norm
    over the complete field. Forecast output steps are scored separately and
    then summed.

    When variables belong to the norm, the joint score is repeated for every
    selected variable for diagnostics. Each repeated value represents the same
    joint score, not an individual variable score.
    """

    def __init__(
        self,
        fair: bool = True,
        norm_over: EnergyScoreNorm = "spatial",
        no_autocast: bool = True,
        ignore_nans: bool = False,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans)
        if norm_over not in ("spatial", "variables", "spatial_and_variables"):
            msg = (
                f"Unknown energy score norm {norm_over!r}. Expected one of: "
                "'spatial', 'variables', 'spatial_and_variables'."
            )
            raise ValueError(msg)
        self.fair = fair
        self.norm_over = norm_over
        self.no_autocast = no_autocast
        self.supports_sharding = True

    @property
    def norm_dimensions(self) -> tuple[int, ...]:
        """Return the tensor dimensions that form each multivariate outcome."""
        if self.norm_over == "spatial":
            return (int(TensorDim.GRID),)
        if self.norm_over == "variables":
            return (int(TensorDim.VARIABLE),)
        return (int(TensorDim.GRID), int(TensorDim.VARIABLE))

    @property
    def variables_are_joint(self) -> bool:
        """Whether variables belong to the energy score norm."""
        return int(TensorDim.VARIABLE) in self.norm_dimensions

    @property
    def name(self) -> str:
        prefix = "f" if self.fair else ""
        return f"{prefix}energy_score_{self.norm_over}"

    @property
    def needs_shard_layout_info(self) -> bool:
        """Spatial norms transpose grid sharding into variable sharding."""
        return self.norm_over == "spatial"

    @staticmethod
    def _resolve_scaler_dimensions(dimension: int | tuple[int, ...]) -> tuple[int, ...]:
        dimensions = (dimension,) if isinstance(dimension, int) else dimension
        num_dimensions = int(TensorDim.VARIABLE) + 1
        return tuple(num_dimensions + int(dim) if -num_dimensions <= int(dim) < 0 else int(dim) for dim in dimensions)

    def _uses_scaler_in_norm(self, dimension: int | tuple[int, ...]) -> bool:
        dimensions = self._resolve_scaler_dimensions(dimension)
        return bool(set(dimensions).intersection(self.norm_dimensions))

    @staticmethod
    def _validate_norm_scaler(scaler: torch.Tensor) -> None:
        scaler = torch.as_tensor(scaler)
        # A weighted Euclidean norm requires finite real weights greater than or equal to zero.
        if torch.is_complex(scaler) or not torch.isfinite(scaler).all():
            msg = "Energy score norm weights must be finite real values."
            raise ValueError(msg)
        if torch.any(scaler < 0):
            msg = "Energy score norm weights must be non-negative."
            raise ValueError(msg)

    def add_scaler(
        self,
        dimension: int | tuple[int, ...],
        scaler: torch.Tensor,
        *,
        name: str | None = None,
    ) -> None:
        if self._uses_scaler_in_norm(dimension):
            self._validate_norm_scaler(scaler)
        super().add_scaler(dimension, scaler, name=name)

    def update_scaler(self, name: str, scaler: torch.Tensor, *, override: bool = False) -> None:
        if name in self.scaler.tensors and self._uses_scaler_in_norm(self.scaler.tensors[name][0]):
            self._validate_norm_scaler(scaler)
        super().update_scaler(name, scaler, override=override)

    @staticmethod
    def _validate_input_shapes(pred: torch.Tensor, target: torch.Tensor) -> None:
        if pred.ndim != 5 or target.ndim != 5:
            msg = (
                "EnergyScoreLoss expects prediction and target tensors with shape "
                "(batch, time, ensemble, grid, variable)."
            )
            raise ValueError(msg)
        if target.shape[TensorDim.ENSEMBLE_DIM] != 1:
            msg = "EnergyScoreLoss requires a singleton target ensemble dimension."
            raise ValueError(msg)
        if pred.shape[:2] != target.shape[:2] or pred.shape[3:] != target.shape[3:]:
            msg = f"Prediction and target shapes are incompatible: {tuple(pred.shape)} and {tuple(target.shape)}."
            raise ValueError(msg)
        if pred.shape[TensorDim.ENSEMBLE_DIM] <= 1:
            msg = "EnergyScoreLoss requires at least two ensemble members."
            raise ValueError(msg)

    def _filtered_scaler(
        self,
        without_scalers: list[str] | list[int] | None,
    ) -> ScaleTensor:
        if not without_scalers:
            return self.scaler
        if isinstance(without_scalers[0], str):
            return self.scaler.without(without_scalers)
        return self.scaler.without_by_dim(without_scalers)

    def _partition_scalers(self, scale_tensor: ScaleTensor) -> tuple[ScaleTensor, ScaleTensor]:
        norm_scalers = {}
        outer_scalers = {}
        norm_dimensions = set(self.norm_dimensions)

        for name, (dimensions, scaler) in scale_tensor.tensors.items():
            resolved_dimensions = self._resolve_scaler_dimensions(dimensions)
            destination = norm_scalers if norm_dimensions.intersection(resolved_dimensions) else outer_scalers
            destination[name] = (dimensions, scaler)

        return ScaleTensor(**norm_scalers), ScaleTensor(**outer_scalers)

    def _select_inputs_and_weights(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        scaler_indices: tuple[int, ...] | None,
        without_scalers: list[str] | list[int] | None,
        grid_shard_slice: slice | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_tensor = self._filtered_scaler(without_scalers)
        norm_scalers, outer_scalers = self._partition_scalers(scale_tensor)
        ones = torch.ones_like(target)
        norm_weights = norm_scalers.scale_iteratively(
            ones,
            subset_indices=scaler_indices,
            grid_shard_slice=grid_shard_slice,
        )
        outer_shape = list(target.shape)
        if int(TensorDim.GRID) in self.norm_dimensions:
            outer_shape[TensorDim.GRID] = 1
        outer_weights = outer_scalers.scale_iteratively(
            target.new_ones(outer_shape),
            subset_indices=scaler_indices,
            grid_shard_slice=grid_shard_slice,
        )

        if scaler_indices is not None:
            if not isinstance(scaler_indices, tuple):
                msg = "scaler_indices must be a tuple of per-dimension indexers, e.g. (..., indices)"
                raise TypeError(msg)
            pred = pred[scaler_indices]
            target = target[scaler_indices]

        return pred, target, norm_weights, outer_weights

    @staticmethod
    def _transpose_grid_to_variables(
        pred: torch.Tensor,
        target: torch.Tensor,
        norm_weights: torch.Tensor,
        group: ProcessGroup,
        grid_dim: int,
        grid_shard_sizes: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        pred_variable_sizes = get_shard_sizes(pred, TensorDim.VARIABLE, group)
        target_variable_sizes = get_shard_sizes(target, TensorDim.VARIABLE, group)
        if pred_variable_sizes != target_variable_sizes:
            msg = (
                "Prediction and target variable shard sizes must match for the spatial energy score: "
                f"{pred_variable_sizes} != {target_variable_sizes}"
            )
            raise ValueError(msg)

        pred = all_to_all_transpose(
            pred,
            TensorDim.VARIABLE,
            pred_variable_sizes,
            grid_dim,
            grid_shard_sizes,
            group,
        )
        target = all_to_all_transpose(
            target,
            TensorDim.VARIABLE,
            target_variable_sizes,
            grid_dim,
            grid_shard_sizes,
            group,
        )
        norm_weights = all_to_all_transpose(
            norm_weights,
            TensorDim.VARIABLE,
            target_variable_sizes,
            grid_dim,
            grid_shard_sizes,
            group,
        )
        return pred, target, norm_weights, target_variable_sizes

    @staticmethod
    def _maximum_across_group(value: torch.Tensor, group: ProcessGroup) -> torch.Tensor:
        # Treat the shared maximum as a constant. It only sets a numerical scale
        # that cancels from max * sqrt(sum((value / max) ** 2)). This preserves
        # the norm gradient and avoids autograd through the distributed maximum.
        maximum = value.detach().clone()
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
        return maximum

    def _weighted_norm(
        self,
        values: torch.Tensor,
        weights: torch.Tensor,
        feature_valid: torch.Tensor | None,
        norm_dimensions: tuple[int, ...],
        group: ProcessGroup | None,
    ) -> torch.Tensor:
        active = (weights > 0).expand_as(values)

        if feature_valid is not None:
            valid = feature_valid.unsqueeze(TensorDim.ENSEMBLE_DIM).expand_as(values)
            active = active & valid

        safe_values = torch.where(active, values, torch.zeros_like(values))
        weighted_abs = torch.abs(safe_values) * torch.sqrt(weights)

        distance_max = weighted_abs.amax(dim=norm_dimensions, keepdim=True)
        if group is not None:
            # A distributed norm uses one largest magnitude across all shards.
            distance_max = self._maximum_across_group(distance_max, group)

        safe_distance_max = torch.where(distance_max > 0, distance_max, torch.ones_like(distance_max))
        scaled_abs = weighted_abs / safe_distance_max
        scaled_abs = torch.where(distance_max > 0, scaled_abs, torch.zeros_like(scaled_abs))

        norm_squared = scaled_abs.square().sum(dim=norm_dimensions)
        if group is not None:
            norm_squared = reduce_tensor(norm_squared, group)

        positive_norm = norm_squared > 0
        safe_norm_squared = torch.where(positive_norm, norm_squared, torch.ones_like(norm_squared))
        norm = torch.where(
            positive_norm,
            torch.sqrt(safe_norm_squared),
            torch.zeros_like(norm_squared),
        )

        for dimension in sorted(norm_dimensions, reverse=True):
            distance_max = distance_max.squeeze(dimension)
        norm = distance_max * norm

        if feature_valid is not None:
            support = active.to(dtype=values.dtype).sum(dim=norm_dimensions)
            if group is not None:
                support = reduce_tensor(support, group)
            norm = norm.masked_fill(support <= 0, torch.nan)

        return norm

    def _score_field(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        norm_weights: torch.Tensor,
        group: ProcessGroup | None,
    ) -> torch.Tensor:
        feature_valid = None
        if self.ignore_nans:
            feature_valid = torch.isfinite(target.squeeze(TensorDim.ENSEMBLE_DIM)) & torch.isfinite(pred).all(
                dim=TensorDim.ENSEMBLE_DIM,
            )

        observation_distances = pred - target
        observation_term = self._weighted_norm(
            observation_distances,
            norm_weights,
            feature_valid,
            self.norm_dimensions,
            group,
        ).mean(dim=TensorDim.ENSEMBLE_DIM)

        ensemble_size = pred.shape[TensorDim.ENSEMBLE_DIM]
        pair_distance_sum = torch.zeros_like(observation_term)
        for member in range(ensemble_size - 1):
            pair_distances = pred[:, :, member].unsqueeze(TensorDim.ENSEMBLE_DIM) - pred[:, :, member + 1 :]
            pair_distance_sum = pair_distance_sum + self._weighted_norm(
                pair_distances,
                norm_weights,
                feature_valid,
                self.norm_dimensions,
                group,
            ).sum(dim=TensorDim.ENSEMBLE_DIM)

        pair_coefficient = 1.0 / (ensemble_size * (ensemble_size - 1)) if self.fair else 1.0 / (ensemble_size**2)
        return observation_term - pair_coefficient * pair_distance_sum

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
        grid_dim: int | None = None,
        grid_shard_sizes: ShardSizes = None,
        squash_mode: Squash_mode = "avg",
        **_kwargs,
    ) -> torch.Tensor:
        """Calculate the energy score over the selected dimensions."""
        self._validate_input_shapes(pred, target)
        if self.variables_are_joint and squash and squash_mode == "sum":
            msg = "squash_mode='sum' is not defined when variables are part of the joint energy score."
            raise ValueError(msg)

        pred, target, norm_weights, outer_weights = self._select_inputs_and_weights(
            pred,
            target,
            scaler_indices,
            without_scalers,
            grid_shard_slice,
        )
        num_variables = pred.shape[TensorDim.VARIABLE]

        is_sharded = grid_shard_slice is not None
        variable_shard_sizes = None
        if is_sharded:
            if group is None:
                msg = "EnergyScoreLoss requires a process group for spatially sharded inputs."
                raise ValueError(msg)
            if self.norm_over == "spatial" and (grid_dim is None or grid_shard_sizes is None):
                msg = (
                    "EnergyScoreLoss with norm_over='spatial' requires grid_dim and "
                    "grid_shard_sizes for spatially sharded inputs."
                )
                raise ValueError(msg)
            if self.norm_over == "spatial":
                assert grid_dim is not None
                assert grid_shard_sizes is not None
                pred, target, norm_weights, variable_shard_sizes = self._transpose_grid_to_variables(
                    pred,
                    target,
                    norm_weights,
                    group,
                    grid_dim,
                    grid_shard_sizes,
                )

        norm_group = group if is_sharded and self.norm_over == "spatial_and_variables" else None
        context = torch.amp.autocast(device_type=pred.device.type, enabled=False) if self.no_autocast else nullcontext()
        with context:
            score = self._score_field(pred, target, norm_weights, norm_group)

        if is_sharded and self.norm_over == "spatial":
            assert group is not None
            assert variable_shard_sizes is not None
            score = gather_tensor(score, -1, variable_shard_sizes, group)

        if self.variables_are_joint:
            # Show the joint variable score beside each selected variable.
            score = score.unsqueeze(-1).expand(*score.shape, num_variables)

        score = score.unsqueeze(TensorDim.ENSEMBLE_DIM)
        if int(TensorDim.GRID) in self.norm_dimensions:
            score = score.unsqueeze(TensorDim.GRID)
        score = score * outer_weights
        if self.ignore_nans:
            score = torch.where(torch.isnan(score), torch.zeros_like(score), score)

        reduction_group = group if is_sharded and self.norm_over == "variables" else None
        return self.reduce(score, squash=squash, squash_mode=squash_mode, group=reduction_group)
