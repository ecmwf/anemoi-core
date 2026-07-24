# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Common execution lifecycle for graph score losses.

Tensor shapes use ``B`` for batch, ``T`` for time, ``M`` for ensemble
members, ``N`` for nodes, and ``V`` for variables:

- ensemble predictions: ``(B, T, M, N, V)``
- targets: ``(B, T, 1, N, V)``
- local score fields: ``(B, T, N, V)``

Concrete losses implement only the local score kernel. This base class handles
input validation, precision control, distributed layout changes, scaling, and
the final reduction around that kernel.
"""

from abc import abstractmethod

import einops
import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import all_to_all_transpose
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import Squash_mode
from anemoi.training.losses.graph_score_graph import GraphScoreGraph
from anemoi.training.utils.enums import TensorDim


class BaseGraphScoreLoss(BaseLoss):
    """Run a local graph score kernel within the standard loss lifecycle."""

    needs_graph_data: bool = True
    graph: GraphScoreGraph | None

    def __init__(
        self,
        *,
        graph: GraphScoreGraph | None,
        no_autocast: bool = True,
        ignore_nans: bool = False,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans)
        self.graph = graph
        self.no_autocast = no_autocast
        self.supports_sharding = True

    @property
    def row_normalize(self) -> bool:
        return self.graph.row_normalize if self.graph is not None else False

    def compile_for_training(self, **options) -> None:
        """Compile only the local graph score kernel."""
        self._compute_local_score_tensor = torch.compile(
            self._compute_local_score_tensor,
            **options,
        )

    @property
    def needs_shard_layout_info(self) -> bool:
        """Whether this loss needs shard layout metadata from the task."""
        return self.graph is not None

    def _prepare_for_aggregation(
        self,
        y_pred_ens: torch.Tensor,
        y: torch.Tensor,
        group: ProcessGroup,
        grid_dim: int,
        grid_shard_sizes: ShardSizes,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Move grid-sharded inputs to a full-grid, variable-sharded layout."""
        channel_shard_sizes_pred = get_shard_sizes(
            y_pred_ens,
            TensorDim.VARIABLE,
            group,
        )
        channel_shard_sizes_target = get_shard_sizes(y, TensorDim.VARIABLE, group)
        if channel_shard_sizes_pred != channel_shard_sizes_target:
            msg = (
                "Prediction and target variable shard sizes must match for graph score losses: "
                f"{channel_shard_sizes_pred} != {channel_shard_sizes_target}"
            )
            raise ValueError(msg)

        y_pred_ens_full = all_to_all_transpose(
            y_pred_ens,
            TensorDim.VARIABLE,
            channel_shard_sizes_pred,
            grid_dim,
            grid_shard_sizes,
            group,
        )
        y_full = all_to_all_transpose(
            y,
            TensorDim.VARIABLE,
            channel_shard_sizes_target,
            grid_dim,
            grid_shard_sizes,
            group,
        )

        return y_pred_ens_full, y_full, channel_shard_sizes_target

    @staticmethod
    def _restore_grid_sharding(
        score: torch.Tensor,
        group: ProcessGroup,
        grid_shard_sizes: list[int],
        channel_shard_sizes: list[int],
    ) -> torch.Tensor:
        """Move a full-grid, variable-sharded score back to grid-sharded layout."""
        return all_to_all_transpose(
            score,
            -2,
            grid_shard_sizes,
            -1,
            channel_shard_sizes,
            group,
        )

    @staticmethod
    def _validate_input_shapes(y_pred_ens: torch.Tensor, y: torch.Tensor) -> None:
        if y_pred_ens.ndim != 5 or y.ndim != 5:
            msg = (
                "Graph score losses expect prediction and target tensors with shape "
                "(batch, time, ensemble, grid, variable)."
            )
            raise ValueError(msg)
        if y.shape[TensorDim.ENSEMBLE_DIM] != 1:
            msg = "Graph score losses require a singleton target ensemble dimension."
            raise ValueError(msg)
        if y_pred_ens.shape[:2] != y.shape[:2] or y_pred_ens.shape[3:] != y.shape[3:]:
            msg = f"Prediction and target shapes are incompatible: {tuple(y_pred_ens.shape)} and {tuple(y.shape)}."
            raise ValueError(msg)

    def _validate_graph_grid_size(self, y_pred_ens: torch.Tensor) -> None:
        if self.graph is None:
            return

        grid_size = y_pred_ens.shape[TensorDim.GRID]
        expected_shape = (grid_size, grid_size)
        if self.graph.shape != expected_shape:
            msg = (
                f"{self.__class__.__name__} loss graph shape {self.graph.shape} does not match "
                f"the forecast grid shape {expected_shape}."
            )
            raise ValueError(msg)

    def _compute_local_score_tensor_with_precision_control(
        self,
        y_pred_ens: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        if not self.no_autocast:
            return self._compute_local_score_tensor(y_pred_ens, y)

        with torch.amp.autocast(device_type=y_pred_ens.device.type, enabled=False):
            return self._compute_local_score_tensor(y_pred_ens, y)

    @abstractmethod
    def _compute_local_score_tensor(
        self,
        y_pred_ens: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a ``(B, T, N, V)`` score from predictions and squeezed targets.

        ``y_pred_ens`` has shape ``(B, T, M, N, V)`` and ``y`` has shape
        ``(B, T, N, V)``. Implementations must not perform collectives,
        scaling, or final time/grid reduction.
        """

    def _format_and_scale_score(
        self,
        score: torch.Tensor,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
    ) -> torch.Tensor:
        score = einops.rearrange(score, "bs t latlon v -> bs t 1 latlon v")
        return self.scale(
            score,
            scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )

    def _score_tensor(
        self,
        y_pred_ens: torch.Tensor,
        y: torch.Tensor,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        grid_dim: int | None = None,
        grid_shard_sizes: ShardSizes = None,
    ) -> tuple[torch.Tensor, bool]:
        # 1. Validate the public five-dimensional prediction/target contract.
        self._validate_input_shapes(y_pred_ens, y)
        assert y_pred_ens.shape[2] > 1, "Ensemble size must be greater than 1."

        is_sharded = grid_shard_slice is not None
        is_model_sharded = self.graph is not None and is_sharded

        # 2. Graph aggregation needs the complete node dimension. For a
        # sharded model, transpose grid shards into variable shards first.
        pred_for_score, target_for_score = y_pred_ens, y
        channel_shard_sizes = None
        if is_model_sharded:
            if group is None:
                msg = f"{self.__class__.__name__} requires a process group for graph-based sharded inputs."
                raise ValueError(msg)
            if grid_dim is None or grid_shard_sizes is None:
                msg = (
                    f"grid_dim and grid_shard_sizes must be provided when {self.__class__.__name__} "
                    "receives graph-based sharded inputs."
                )
                raise ValueError(msg)
            pred_for_score, target_for_score, channel_shard_sizes = self._prepare_for_aggregation(
                y_pred_ens,
                y,
                group,
                grid_dim,
                grid_shard_sizes,
            )

        # 3. Remove the target's singleton ensemble dimension and evaluate the
        # concrete score formula on the full graph node space.
        self._validate_graph_grid_size(pred_for_score)
        target_for_score = target_for_score.squeeze(TensorDim.ENSEMBLE_DIM)
        score = self._compute_local_score_tensor_with_precision_control(
            pred_for_score,
            target_for_score,
        )

        # 4. Restore the original grid-sharded layout before applying the
        # standard node/variable scalers.
        if is_model_sharded:
            assert grid_shard_sizes is not None
            assert channel_shard_sizes is not None
            score = self._restore_grid_sharding(
                score,
                group,
                grid_shard_sizes,
                channel_shard_sizes,
            )

        score = self._format_and_scale_score(
            score,
            scaler_indices=scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )
        return score, is_sharded

    def local_score_field(
        self,
        y_pred_ens: torch.Tensor,
        y: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
    ) -> torch.Tensor:
        """Return the local graph score field before time/grid reduction."""
        self._validate_input_shapes(y_pred_ens, y)
        self._validate_graph_grid_size(y_pred_ens)
        y = y.squeeze(TensorDim.ENSEMBLE_DIM)
        score = self._compute_local_score_tensor_with_precision_control(y_pred_ens, y)
        score = self._format_and_scale_score(
            score,
            scaler_indices=scaler_indices,
            without_scalers=without_scalers,
        )
        if squash:
            score = torch.nansum(score, dim=-1) if self.ignore_nans else torch.sum(score, dim=-1)
        avg_function = torch.nanmean if self.ignore_nans else torch.mean
        return avg_function(score, dim=(0, 2))

    def forward(
        self,
        y_pred_ens: torch.Tensor,
        y: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        grid_dim: int | None = None,
        grid_shard_sizes: ShardSizes = None,
        squash_mode: Squash_mode = "sum",
        **kwargs,  # noqa: ARG002
    ) -> torch.Tensor:
        score, is_sharded = self._score_tensor(
            y_pred_ens,
            y,
            scaler_indices=scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
            group=group,
            grid_dim=grid_dim,
            grid_shard_sizes=grid_shard_sizes,
        )
        # Unsupported rows stay NaN through local scoring and scaling. They are
        # neutralized only for the final distributed reduction.
        if self.ignore_nans:
            score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
        return self.reduce(
            score,
            squash=squash,
            squash_mode=squash_mode,
            group=group if is_sharded else None,
        )
