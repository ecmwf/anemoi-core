# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import abstractmethod
from contextlib import nullcontext

import einops
import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import all_to_all_transpose
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import Squash_mode
from anemoi.training.losses.graph_score_graph import GraphScoreGraph
from anemoi.training.utils.enums import TensorDim


def csr_matmul(matrix: torch.Tensor, node_values: torch.Tensor) -> torch.Tensor:
    """Apply ``matrix`` to the node dimension of ``(..., N, V)`` values."""
    if matrix.layout != torch.sparse_csr:
        msg = f"csr_matmul requires a CSR matrix, got {matrix.layout}."
        raise TypeError(msg)
    input_shape = node_values.shape
    input_nodes, channels = input_shape[-2:]
    if matrix.shape[1] != input_nodes:
        msg = f"CSR matrix width {matrix.shape[1]} does not match the input node count {input_nodes}."
        raise ValueError(msg)

    dense_batches = node_values.numel() // (input_nodes * channels)
    right_hand_side = (
        node_values.reshape(dense_batches, input_nodes, channels)
        .permute(1, 0, 2)
        .reshape(input_nodes, dense_batches * channels)
    )
    projected = torch.sparse.mm(matrix, right_hand_side)
    output_nodes = projected.shape[0]
    return (
        projected.reshape(output_nodes, dense_batches, channels)
        .permute(1, 0, 2)
        .reshape(*input_shape[:-2], output_nodes, channels)
    )


def scale_node_differences(differences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale along nodes so squares remain representable in float32."""
    scale = differences.abs().amax(dim=-2, keepdim=True)
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return differences / safe_scale, scale


def safe_sqrt(values: torch.Tensor) -> torch.Tensor:
    """Take a square root with a finite zero derivative at zero."""
    positive = values > 0
    safe_values = torch.where(positive, values, torch.ones_like(values))
    return torch.where(positive, torch.sqrt(safe_values), torch.zeros_like(values))


class BaseGraphScoreLoss(BaseLoss):
    """Evaluate a graph score with one shared CSR graph per model grid."""

    needs_graph_data: bool = True
    uses_edge_tensors: bool = False
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

    @property
    def graph_provider(self) -> ProjectionGraphProvider | None:
        """Return the sparse projection provider used by this score."""
        return self.graph.graph_provider if self.graph is not None else None

    def compile_for_training(self, **options) -> None:
        """Compile only the numerical score kernel."""
        self._compute_local_score_tensor = torch.compile(self._compute_local_score_tensor, **options)

    @property
    def needs_shard_layout_info(self) -> bool:
        return self.graph is not None

    def _prepare_for_aggregation(
        self,
        y_pred_ens: torch.Tensor,
        y_target: torch.Tensor,
        group: ProcessGroup,
        grid_dim: int,
        grid_shard_sizes: ShardSizes,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Gather all node values and shard the variable axis."""
        channel_shard_sizes_pred = get_shard_sizes(y_pred_ens, TensorDim.VARIABLE, group)
        channel_shard_sizes_target = get_shard_sizes(y_target, TensorDim.VARIABLE, group)
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
        y_target_full = all_to_all_transpose(
            y_target,
            TensorDim.VARIABLE,
            channel_shard_sizes_target,
            grid_dim,
            grid_shard_sizes,
            group,
        )
        return y_pred_ens_full, y_target_full, channel_shard_sizes_target

    @staticmethod
    def _restore_grid_sharding(
        score: torch.Tensor,
        group: ProcessGroup,
        grid_shard_sizes: list[int],
        channel_shard_sizes: list[int],
    ) -> torch.Tensor:
        return all_to_all_transpose(
            score,
            -2,
            grid_shard_sizes,
            -1,
            channel_shard_sizes,
            group,
        )

    @staticmethod
    def _validate_input_shapes(y_pred_ens: torch.Tensor, y_target: torch.Tensor) -> None:
        if y_pred_ens.ndim != 5 or y_target.ndim != 5:
            msg = (
                "Graph score losses expect prediction and target tensors with shape "
                "(batch, time, ensemble, grid, variable)."
            )
            raise ValueError(msg)
        if y_target.shape[TensorDim.ENSEMBLE_DIM] != 1:
            msg = "Graph score losses require a singleton target ensemble dimension."
            raise ValueError(msg)
        if y_pred_ens.shape[:2] != y_target.shape[:2] or y_pred_ens.shape[3:] != y_target.shape[3:]:
            msg = (
                f"Prediction and target shapes are incompatible: "
                f"{tuple(y_pred_ens.shape)} and {tuple(y_target.shape)}."
            )
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

    @staticmethod
    def _align_input_dtypes(
        y_pred_ens: torch.Tensor,
        y_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Ensure that predictions and targets have the correct dtype."""
        allowed_dtypes = (torch.float32, torch.float64)
        if y_pred_ens.dtype not in allowed_dtypes or y_target.dtype not in allowed_dtypes:
            msg = (
                "Graph score inputs must be float32 or float64, "
                f"but received prediction dtype {y_pred_ens.dtype} "
                f"and target dtype {y_target.dtype}."
            )
            raise TypeError(msg)

        score_dtype = torch.promote_types(y_pred_ens.dtype, y_target.dtype)
        return (
            y_pred_ens.to(dtype=score_dtype),
            y_target.to(dtype=score_dtype),
        )

    def _graph_kernel_tensors(
        self,
        reference: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Prepare CSR tensors eagerly for the compiled numerical kernel."""
        if self.graph is None:
            return None, None, None, None
        if self.uses_edge_tensors:
            source_index, destination_index, edge_weights = self.graph.get_edge_tensors(
                device=reference.device,
                dtype=reference.dtype,
            )
            # Use src / dst idx and edge weights to allow Dynamo/Inductor fusion.
            return None, source_index, destination_index, edge_weights
        matrix = self.graph.get_matrix(device=reference.device, dtype=reference.dtype)
        return matrix, None, None, None

    @abstractmethod
    def _compute_local_score_tensor(
        self,
        y_pred_ens: torch.Tensor,
        y_target: torch.Tensor,
        matrix: torch.Tensor | None,
        source_index: torch.Tensor | None,
        destination_index: torch.Tensor | None,
        edge_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return local scores with shape ``(B, T, N, V)``."""

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
        y_target: torch.Tensor,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        grid_dim: int | None = None,
        grid_shard_sizes: ShardSizes = None,
    ) -> tuple[torch.Tensor, bool]:
        self._validate_input_shapes(y_pred_ens, y_target)
        assert y_pred_ens.shape[TensorDim.ENSEMBLE_DIM] > 1, "Ensemble size must be greater than 1."

        is_sharded = grid_shard_slice is not None
        is_model_sharded = self.graph is not None and is_sharded
        pred_for_score, target_for_score = y_pred_ens, y_target
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
                y_target,
                group,
                grid_dim,
                grid_shard_sizes,
            )

        self._validate_graph_grid_size(pred_for_score)
        target_for_score = target_for_score.squeeze(TensorDim.ENSEMBLE_DIM)
        pred_for_score, target_for_score = self._align_input_dtypes(
            pred_for_score,
            target_for_score,
        )
        graph_tensors = self._graph_kernel_tensors(pred_for_score)
        context = (
            torch.amp.autocast(device_type=pred_for_score.device.type, enabled=False)
            if self.no_autocast
            else nullcontext()
        )
        with context:
            score = self._compute_local_score_tensor(
                pred_for_score,
                target_for_score,
                *graph_tensors,
            )

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
        y_target: torch.Tensor,
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
            y_target,
            scaler_indices=scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
            group=group,
            grid_dim=grid_dim,
            grid_shard_sizes=grid_shard_sizes,
        )
        if self.ignore_nans:
            score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
        return self.reduce(
            score,
            squash=squash,
            squash_mode=squash_mode,
            group=group if is_sharded else None,
        )
