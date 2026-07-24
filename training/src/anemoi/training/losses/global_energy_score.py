# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Energy scores over the full forecast grid."""

from contextlib import nullcontext

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


class GlobalEnergyScoreLoss(BaseLoss):
    """Energy score over the full spatial field.

    By default, the spatial norm is calculated separately for each variable.
    With ``joint_variables=True``, space and variables belong to one joint
    norm. Forecast output steps are scored separately and then summed.

    For diagnostics, the joint score is repeated for every selected variable.
    These repeated values all describe the same joint field rather than
    separate single-variable scores.
    """

    def __init__(
        self,
        fair: bool = True,
        joint_variables: bool = False,
        no_autocast: bool = True,
        ignore_nans: bool = False,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans)
        self.fair = fair
        self.joint_variables = joint_variables
        self.no_autocast = no_autocast
        self.supports_sharding = True

    @property
    def name(self) -> str:
        prefix = "f" if self.fair else ""
        return f"{prefix}global_energy_score"

    @property
    def needs_shard_layout_info(self) -> bool:
        """The distance requires the complete spatial field."""
        return True

    @staticmethod
    def _resolve_scaler_dimensions(dimension: int | tuple[int, ...]) -> tuple[int, ...]:
        dimensions = (dimension,) if isinstance(dimension, int) else dimension
        num_dimensions = int(TensorDim.VARIABLE) + 1
        return tuple(num_dimensions + int(dim) if -num_dimensions <= int(dim) < 0 else int(dim) for dim in dimensions)

    def _uses_scaler_in_norm(self, dimension: int | tuple[int, ...]) -> bool:
        dimensions = self._resolve_scaler_dimensions(dimension)
        return TensorDim.GRID in dimensions or (self.joint_variables and TensorDim.VARIABLE in dimensions)

    @staticmethod
    def _validate_norm_scaler(scaler: torch.Tensor) -> None:
        scaler = torch.as_tensor(scaler)
        # A weighted Euclidean norm requires finite, real, non-negative weights.
        if torch.is_complex(scaler) or not torch.isfinite(scaler).all():
            msg = "Global energy score weights must be finite real values."
            raise ValueError(msg)
        if torch.any(scaler < 0):
            msg = "Global energy score weights must be non-negative."
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
                "GlobalEnergyScoreLoss expects prediction and target tensors with shape "
                "(batch, time, ensemble, grid, variable)."
            )
            raise ValueError(msg)
        if target.shape[TensorDim.ENSEMBLE_DIM] != 1:
            msg = "GlobalEnergyScoreLoss requires a singleton target ensemble dimension."
            raise ValueError(msg)
        if pred.shape[:2] != target.shape[:2] or pred.shape[3:] != target.shape[3:]:
            msg = f"Prediction and target shapes are incompatible: {tuple(pred.shape)} and {tuple(target.shape)}."
            raise ValueError(msg)
        if pred.shape[TensorDim.ENSEMBLE_DIM] <= 1:
            msg = "GlobalEnergyScoreLoss requires at least two ensemble members."
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
        norm_dimensions = {TensorDim.GRID}
        if self.joint_variables:
            norm_dimensions.add(TensorDim.VARIABLE)

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
        outer_shape[TensorDim.GRID] = 1
        outer_weights = outer_scalers.scale_iteratively(
            target.new_ones(outer_shape),
            subset_indices=scaler_indices,
            grid_shard_slice=None,
        )

        if scaler_indices is not None:
            if not isinstance(scaler_indices, tuple):
                msg = "scaler_indices must be a tuple of per-dimension indexers, e.g. (..., indices)"
                raise TypeError(msg)
            pred = pred[scaler_indices]
            target = target[scaler_indices]

        return pred, target, norm_weights, outer_weights

    @staticmethod
    def _prepare_for_aggregation(
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
                "Prediction and target variable shard sizes must match for the global energy score: "
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
            # The joint norm uses one largest magnitude across all variables.
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

        norm_dimensions = (int(TensorDim.GRID),)
        if self.joint_variables:
            norm_dimensions = (int(TensorDim.GRID), int(TensorDim.VARIABLE))

        observation_distances = pred - target
        observation_term = self._weighted_norm(
            observation_distances,
            norm_weights,
            feature_valid,
            norm_dimensions,
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
                norm_dimensions,
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
        """Calculate the energy score over space or over space and variables."""
        self._validate_input_shapes(pred, target)
        if self.joint_variables and squash and squash_mode == "sum":
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
            if group is None or grid_dim is None or grid_shard_sizes is None:
                msg = (
                    "GlobalEnergyScoreLoss requires group, grid_dim, and grid_shard_sizes "
                    "for spatially sharded inputs."
                )
                raise ValueError(msg)
            pred, target, norm_weights, variable_shard_sizes = self._prepare_for_aggregation(
                pred,
                target,
                norm_weights,
                group,
                grid_dim,
                grid_shard_sizes,
            )

        norm_group = group if is_sharded and self.joint_variables else None
        context = torch.amp.autocast(device_type=pred.device.type, enabled=False) if self.no_autocast else nullcontext()
        with context:
            score = self._score_field(pred, target, norm_weights, norm_group)

        if is_sharded and not self.joint_variables:
            assert group is not None
            assert variable_shard_sizes is not None
            score = gather_tensor(score, -1, variable_shard_sizes, group)

        if self.joint_variables:
            # The joint field has one score, shown beside each selected variable.
            score = score.unsqueeze(-1).expand(*score.shape, num_variables)

        score = score.unsqueeze(TensorDim.ENSEMBLE_DIM).unsqueeze(TensorDim.GRID)
        score = score * outer_weights[..., :1, :]
        if self.ignore_nans:
            score = torch.where(torch.isnan(score), torch.zeros_like(score), score)

        return self.reduce(score, squash=squash, squash_mode=squash_mode)
