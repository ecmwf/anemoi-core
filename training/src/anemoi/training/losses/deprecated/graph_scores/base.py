# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Common shapes, weights, and sums used by graph scores.

Shapes use ``B`` for batch, ``T`` for forecast output steps, ``M`` for
ensemble members, ``N`` for nodes, and ``V`` for variables:

- ensemble predictions: ``(B, T, M, N, V)``
- targets: ``(B, T, 1, N, V)``
- scores for each output step, node, and variable: ``(B, T, N, V)``

Graph-backed kernels flatten batch and nodes to operate on the batched
disjoint graph returned by the graph provider:

- ensemble predictions: ``(T, M, B*N, V)``
- targets: ``(T, B*N, V)``
- local scores: ``(T, B*N, V)``

Each forecast output step is scored independently. The configured weights are
then applied, the values are summed over output steps and nodes, and variables
are averaged unless a summation is requested.
"""

from abc import abstractmethod
from contextlib import nullcontext

import einops
import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import all_to_all_transpose
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.graph_provider import StaticGraphProvider
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import Squash_mode
from anemoi.training.losses.deprecated.graph_scores.graph import LegacyGraphScoreGraph
from anemoi.training.utils.enums import TensorDim


class LegacyBaseGraphScoreLoss(BaseLoss):
    """Evaluate a graph score and apply its weights and sums."""

    needs_graph_data: bool = True
    graph: LegacyGraphScoreGraph | None

    def __init__(
        self,
        *,
        graph: LegacyGraphScoreGraph | None,
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

    @property
    def graph_provider(self) -> StaticGraphProvider | None:
        """Return the graph provider used by graph-backed scores."""
        return self.graph.graph_provider if self.graph is not None else None

    def compile_for_training(self, **options) -> None:
        """Compile the score calculation used during training."""
        self._compute_local_score_tensor = torch.compile(
            self._compute_local_score_tensor,
            **options,
        )

    @property
    def needs_shard_layout_info(self) -> bool:
        """Return whether a graph is used."""
        return self.graph is not None

    def _prepare_for_aggregation(
        self,
        y_pred_ens: torch.Tensor,
        y: torch.Tensor,
        group: ProcessGroup,
        grid_dim: int,
        grid_shard_sizes: ShardSizes,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Bring values from all nodes together."""
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
        """Return each score to the node on which it originated."""
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

    def _get_batched_graph(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return batch-expanded edges and scalar weights without sharding."""
        if self.graph_provider is None:
            return None, None

        edge_attributes, edge_index, edge_shard_sizes = self.graph_provider.get_edges(
            batch_size=batch_size,
            shard_edges=False,
            act_checkpoint=False,
        )
        assert edge_attributes is not None
        assert edge_index is not None
        assert edge_shard_sizes is None
        assert edge_attributes.shape[-1] == 1, "Graph score providers require one scalar edge weight."
        return edge_index, edge_attributes[:, 0]

    @abstractmethod
    def _compute_local_score_tensor(
        self,
        y_pred_ens: torch.Tensor,
        y: torch.Tensor,
        edge_index: torch.Tensor | None,
        edge_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return the flattened score ``S[T, B*N, V]`` before weights and sums.

        ``y_pred_ens`` has shape ``(T, M, B*N, V)`` and ``y`` has shape
        ``(T, B*N, V)``. Graph-backed scores receive batch-expanded edges;
        the pointwise graph energy score receives ``None`` for both graph
        tensors. No weights or sums over ``T`` and ``N`` are applied here.
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
        # Predictions and observations must cover the same output steps, nodes,
        # and variables. At least two ensemble members are needed.
        self._validate_input_shapes(y_pred_ens, y)
        assert y_pred_ens.shape[2] > 1, "Ensemble size must be greater than 1."

        is_sharded = grid_shard_slice is not None
        is_model_sharded = self.graph is not None and is_sharded

        # Bring neighbouring node values together before measuring graph
        # distances.
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

        # Expand the graph across the batch, then flatten batch and nodes into
        # the provider's disjoint-graph node space. Provider execution and
        # reshaping remain eager; only the numerical score kernel is compiled.
        self._validate_graph_grid_size(pred_for_score)
        batch_size = pred_for_score.shape[TensorDim.BATCH_SIZE]
        num_nodes = pred_for_score.shape[TensorDim.GRID]
        edge_index, edge_weights = self._get_batched_graph(batch_size)

        target_for_score = target_for_score.squeeze(TensorDim.ENSEMBLE_DIM)
        pred_for_score = einops.rearrange(
            pred_for_score,
            "bs t ensemble latlon v -> t ensemble (bs latlon) v",
        )
        target_for_score = einops.rearrange(
            target_for_score,
            "bs t latlon v -> t (bs latlon) v",
        )
        context = (
            torch.amp.autocast(device_type=pred_for_score.device.type, enabled=False)
            if self.no_autocast
            else nullcontext()
        )
        with context:
            score = self._compute_local_score_tensor(
                pred_for_score,
                target_for_score,
                edge_index,
                edge_weights,
            )

        score = einops.rearrange(
            score,
            "t (bs latlon) v -> bs t latlon v",
            bs=batch_size,
            latlon=num_nodes,
        )

        # Return scores to their original nodes, then apply node and variable
        # weights.
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
        squash_mode: Squash_mode = "avg",
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
        # Neighbourhoods with no available values contribute zero to the final
        # sums.
        if self.ignore_nans:
            score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
        return self.reduce(
            score,
            squash=squash,
            squash_mode=squash_mode,
            group=group if is_sharded else None,
        )
