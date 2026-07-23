# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from abc import abstractmethod
from collections.abc import Mapping

import einops
import torch
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import all_to_all_transpose
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import Squash_mode
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


class GraphScoreGraph(nn.Module):
    """Static graph metadata shared by graph score losses."""

    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor,
        *,
        num_src_nodes: int,
        num_dst_nodes: int,
        row_normalize: bool,
    ) -> None:
        super().__init__()
        self.register_buffer("edge_index", edge_index.long(), persistent=False)
        self.register_buffer("edge_src_index", edge_index[0].long(), persistent=False)
        self.register_buffer("edge_dst_index", edge_index[1].long(), persistent=False)
        self.register_buffer("edge_weights", edge_weights, persistent=False)
        self.num_src_nodes = num_src_nodes
        self.num_dst_nodes = num_dst_nodes
        self.row_normalize = row_normalize

    @property
    def shape(self) -> tuple[int, int]:
        return (self.num_dst_nodes, self.num_src_nodes)

    @property
    def num_nodes(self) -> int:
        return self.num_dst_nodes

    def aggregation_metadata(self, dtype: torch.dtype) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.num_dst_nodes, self.edge_dst_index, self.edge_src_index, self.edge_weights.to(dtype=dtype)

    @classmethod
    def from_definition(
        cls,
        graph_definition: Mapping[str, object] | None,
        graph_data: HeteroData | None,
        *,
        graph_name: str,
        allow_none: bool = False,
        require_square: bool = True,
    ) -> "GraphScoreGraph | None":
        if graph_definition is None:
            if allow_none:
                LOGGER.info("%s: %s", graph_name, None)
                return None
            error_msg = f"{graph_name} must be provided."
            raise AssertionError(error_msg)

        if not isinstance(graph_definition, Mapping):
            msg = f"{graph_name} must be a mapping or None, got {type(graph_definition).__name__}."
            raise TypeError(msg)

        assert graph_data is not None, "graph_data must be provided when using a graph score loss graph."

        edges_name = graph_definition.get("edges_name")
        assert edges_name is not None, "Graph score definition must include 'edges_name'."
        edges_name = tuple(edges_name)

        sub_graph = graph_data[edges_name]
        edge_index = sub_graph.edge_index.long()

        edge_weight_attribute = graph_definition.get("edge_weight_attribute")
        if edge_weight_attribute is not None:
            edge_weights = sub_graph[edge_weight_attribute].reshape(-1)
        else:
            edge_weights = torch.ones(edge_index.shape[1], dtype=torch.float32, device=edge_index.device)

        src_node_weight_attribute = graph_definition.get("src_node_weight_attribute")
        if src_node_weight_attribute is not None:
            src_weights = graph_data[edges_name[0]][src_node_weight_attribute].reshape(-1)
            edge_weights = edge_weights * src_weights[edge_index[0]]

        num_src_nodes = graph_data[edges_name[0]].num_nodes
        num_dst_nodes = graph_data[edges_name[2]].num_nodes
        cls._validate_node_space(
            edges_name,
            num_src_nodes,
            num_dst_nodes,
            require_square=require_square,
        )

        cls._validate_weights(
            edge_index[1].long(),
            edge_weights,
            num_dst_nodes,
            graph_name=graph_name,
        )

        row_normalize = bool(graph_definition.get("row_normalize", False))
        if row_normalize:
            edge_weights = cls._row_normalize_weights(edge_index[1].long(), edge_weights, num_dst_nodes)

        cls._validate_row_sums(
            edge_index[1].long(),
            edge_weights,
            num_dst_nodes,
            graph_definition.get("validate_row_sums", True),
            graph_name=graph_name,
        )

        if require_square and num_src_nodes == num_dst_nodes:
            LOGGER.info("%s: edges=%s nodes=%s", graph_name, edge_index.shape[1], num_dst_nodes)
        else:
            LOGGER.info(
                "%s: edges=%s src_nodes=%s dst_nodes=%s",
                graph_name,
                edge_index.shape[1],
                num_src_nodes,
                num_dst_nodes,
            )

        return cls(
            edge_index=edge_index,
            edge_weights=edge_weights,
            num_src_nodes=num_src_nodes,
            num_dst_nodes=num_dst_nodes,
            row_normalize=row_normalize,
        )

    @staticmethod
    def _validate_node_space(
        edges_name: tuple[str, ...],
        num_src_nodes: int,
        num_dst_nodes: int,
        *,
        require_square: bool,
    ) -> None:
        if not require_square:
            return
        if edges_name[0] != edges_name[2]:
            msg = (
                "Graph score losses require source and destination nodes to use the same node type, "
                f"got {edges_name[0]!r} and {edges_name[2]!r}."
            )
            raise ValueError(msg)
        if num_src_nodes != num_dst_nodes:
            msg = (
                "Graph score losses require a grid-preserving loss graph with the same number "
                "of source and target nodes."
            )
            raise ValueError(msg)

    @staticmethod
    def _row_normalize_weights(row_index: torch.Tensor, weights: torch.Tensor, num_rows: int) -> torch.Tensor:
        totals = torch.zeros(num_rows, dtype=weights.dtype, device=weights.device)
        totals = totals.scatter_add_(0, row_index, weights)
        return weights / totals[row_index]

    @staticmethod
    def _validate_weights(
        row_index: torch.Tensor,
        weights: torch.Tensor,
        num_rows: int,
        *,
        graph_name: str,
    ) -> None:
        if weights.numel() != row_index.numel():
            msg = (
                f"{graph_name} must provide exactly one scalar weight per edge, "
                f"got {weights.numel()} weights for {row_index.numel()} edges."
            )
            raise ValueError(msg)
        if torch.is_complex(weights) or not torch.isfinite(weights).all():
            msg = f"{graph_name} weights must be finite real values."
            raise ValueError(msg)
        if torch.any(weights < 0):
            msg = f"{graph_name} weights must be non-negative."
            raise ValueError(msg)

        row_totals = torch.zeros(num_rows, dtype=weights.dtype, device=weights.device)
        row_totals.scatter_add_(0, row_index, weights)
        zero_weight_rows = torch.count_nonzero(row_totals <= 0).item()
        if zero_weight_rows:
            msg = f"{graph_name} must have positive total weight for every node; found {zero_weight_rows} empty rows."
            raise ValueError(msg)

    @staticmethod
    def _validate_row_sums(
        row_index: torch.Tensor,
        weights: torch.Tensor,
        num_rows: int,
        validate_row_sums: bool,
        *,
        graph_name: str,
    ) -> None:
        if not validate_row_sums:
            return

        row_sums = torch.zeros(num_rows, dtype=weights.dtype, device=weights.device).scatter_add_(0, row_index, weights)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
            LOGGER.warning(
                "%s row weights do not sum to 1 (min=%.4f, max=%.4f, mean=%.4f). "
                "Consider using row_normalize=True or pre-normalized weights.",
                graph_name,
                row_sums.min().item(),
                row_sums.max().item(),
                row_sums.mean().item(),
            )


class BaseGraphScoreLoss(BaseLoss):
    """Shared base for graph score losses."""

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
        self._compute_local_score_tensor = torch.compile(self._compute_local_score_tensor, **options)

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
        channel_shard_sizes_pred = get_shard_sizes(y_pred_ens, TensorDim.VARIABLE, group)
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
    def _compute_local_score_tensor(self, y_pred_ens: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the local graph score tensor without collectives or scaling."""

    def _format_and_scale_score(
        self,
        score: torch.Tensor,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
    ) -> torch.Tensor:
        score = einops.rearrange(score, "bs t latlon v -> bs t 1 latlon v")
        return self.scale(score, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)

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
        self._validate_input_shapes(y_pred_ens, y)
        assert y_pred_ens.shape[2] > 1, "Ensemble size must be greater than 1."

        is_sharded = grid_shard_slice is not None
        is_model_sharded = self.graph is not None and is_sharded

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

        self._validate_graph_grid_size(pred_for_score)
        target_for_score = target_for_score.squeeze(TensorDim.ENSEMBLE_DIM)
        score = self._compute_local_score_tensor_with_precision_control(pred_for_score, target_for_score)

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
        if self.ignore_nans:
            score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
        return self.reduce(score, squash=squash, squash_mode=squash_mode, group=group if is_sharded else None)
