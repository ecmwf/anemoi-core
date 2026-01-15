# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from pathlib import Path
from typing import Any

import einops
import torch
from omegaconf import DictConfig
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.graphs.create import GraphCreator
from anemoi.models.distributed.graph import gather_channels
from anemoi.models.distributed.graph import shard_channels
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.models.layers.sparse_projector import SparseProjector
from anemoi.training.losses.base import BaseLoss

LOGGER = logging.getLogger(__name__)


class MultiscaleLossWrapper(BaseLoss):

    name: str = "MultiscaleLossWrapper"

    def __init__(
        self,
        per_scale_loss: BaseLoss,
        weights: list[float],
        keep_batch_sharded: bool,
        loss_matrices_path: Path | str | None = None,
        loss_matrices: list[Path | str] | None = None,
        autocast: bool = False,
        loss_graphs_path: Path | str | None = None,
        loss_graphs: list[dict[str, Any] | None] | None = None,
    ) -> None:
        """Wrapper for multi-scale loss computation.

        Parameters
        ----------
        per_scale_loss : BaseLoss
            Loss to be used at each scale
        weights : list[float]
            Per-scale loss weights
        keep_batch_sharded : bool
            Whether to keep the batch sharded during loss computation
        loss_matrices_path : Path | str | None
            Path to the directory containing smoothing matrices
        loss_matrices : list[Path | str] | None
            Filenames of the smoothing matrices (must preserve grid size)
        autocast : bool
            Whether to use automatic mixed precision for the projections
        loss_graphs_path : Path | str | None
            Optional base path for loss graph configs
        loss_graphs : list[dict[str, Any] | None] | None
            Graph-based smoothing specs (cannot be combined with loss_matrices)
        """
        super().__init__()

        self.smoothing_matrices = self._load_smoothing_matrices(
            loss_matrices_path,
            loss_matrices,
            loss_graphs_path,
            loss_graphs,
        )
        self.num_scales = len(self.smoothing_matrices)
        assert (
            len(weights) == self.num_scales
        ), f"Number of weights ({len(weights)}) must match number of scales ({self.num_scales})"
        self.weights = weights
        self.loss = per_scale_loss
        self.scaler = self.loss.scaler
        self.keep_batch_sharded = keep_batch_sharded
        self.supports_sharding = True
        self.mloss = None
        self.projector = SparseProjector(autocast=autocast)

    def update_scaler(self, name: str, scaler: torch.Tensor, *, override: bool = False) -> None:
        """Update the scaler values for the internal loss.

        Parameters
        ----------
        name : str
            Name of the scaler to update
        scaler : torch.Tensor
            New scaler values
        override : bool, optional
            Whether to override existing scaler values, by default False
        """
        self.loss.update_scaler(name=name, scaler=scaler, override=override)

    def _load_smoothing_matrices(
        self,
        loss_matrices_path: Path | str | None,
        loss_matrices: list[Path | str] | None,
        loss_graphs_path: Path | str | None,
        loss_graphs: list[dict[str, Any] | None] | None,
    ) -> list[ProjectionGraphProvider | None]:
        """Load smoothing sources for multi-scale loss computation."""
        assert not (
            loss_matrices is not None and loss_graphs is not None
        ), "loss_matrices and loss_graphs are mutually exclusive"

        if loss_graphs is not None:
            return self._load_smoothing_graphs(loss_graphs_path, loss_graphs)

        return self._load_smoothing_matrices_from_files(loss_matrices_path, loss_matrices)

    @staticmethod
    def _resolve_graph_config(graph_config: Any, loss_graphs_path: Path | str | None) -> Any:
        if isinstance(graph_config, (str, Path)):
            graph_path = Path(graph_config)
            if loss_graphs_path and not graph_path.is_absolute():
                graph_path = Path(loss_graphs_path) / graph_path
            return graph_path
        return graph_config

    def _build_loss_graph(self, graph_config: Any, loss_graphs_path: Path | str | None) -> HeteroData:
        resolved_config = self._resolve_graph_config(graph_config, loss_graphs_path)
        graph_creator = GraphCreator(config=resolved_config)
        graph = HeteroData()
        graph = graph_creator.update_graph(graph)
        graph = graph_creator.clean(graph)
        return graph_creator.post_process(graph)

    def _load_smoothing_graphs(
        self,
        loss_graphs_path: Path | str | None,
        loss_graphs: list[dict[str, Any] | None] | None,
    ) -> list[ProjectionGraphProvider | None]:
        """Load smoothing graphs for multi-scale loss computation."""
        if not loss_graphs:
            LOGGER.info("No loss graphs specified, using single scale without smoothing")
            return [None]

        smoothing_matrices: list[ProjectionGraphProvider | None] = []

        for graph_spec in loss_graphs:
            if graph_spec is None or graph_spec is False or graph_spec == "None":
                smoothing_matrices.append(None)
                LOGGER.info("Loss smoothing: %s", None)
                continue

            assert isinstance(graph_spec, (dict, DictConfig)), "loss_graphs entries must be mappings"
            graph_config = graph_spec.get("graph_config")
            assert graph_config is not None, "loss_graphs entries must define graph_config"
            edges_name = graph_spec.get("edges_name")
            assert edges_name is not None, "loss_graphs entries must define edges_name"
            assert len(edges_name) == 3, "edges_name must be [src, relation, dst]"

            edge_weight_attribute = graph_spec.get("edge_weight_attribute")
            row_normalize = bool(graph_spec.get("row_normalize", False))

            graph = self._build_loss_graph(graph_config, loss_graphs_path)
            provider = ProjectionGraphProvider(
                graph=graph,
                edges_name=tuple(edges_name),
                edge_weight_attribute=edge_weight_attribute,
                row_normalize=row_normalize,
            )
            smoothing_matrices.append(provider)
            LOGGER.info("Loss smoothing: %s", provider.get_edges().shape)

        return smoothing_matrices

    def _load_smoothing_matrices_from_files(
        self,
        loss_matrices_path: Path | str | None,
        loss_matrices: list[Path | str] | None,
    ) -> list[ProjectionGraphProvider | None]:
        """Load smoothing matrices for multi-scale loss computation.

        These matrices apply spatial smoothing while preserving grid size.
        """
        smoothing_matrices = []

        # Handle None, empty list, or falsy values - default to single scale with no smoothing
        if not loss_matrices:
            LOGGER.info("No smoothing files specified, using single scale without smoothing")
            return [None]

        for filename in loss_matrices:
            # Skip None, False, or the string "None"
            if filename is None or filename is False or filename == "None":
                smoothing_matrices.append(None)
                LOGGER.info("Loss smoothing: %s", None)
            else:
                file_path = Path(filename)
                if loss_matrices_path and not file_path.is_absolute():
                    file_path = Path(loss_matrices_path) / file_path
                provider = ProjectionGraphProvider(
                    file_path=file_path,
                    row_normalize=False,
                )
                smoothing_matrices.append(provider)
                LOGGER.info("Loss smoothing: %s", provider.get_edges().shape)

        return smoothing_matrices

    def _prepare_for_smoothing(
        self,
        y_pred_ens: torch.Tensor,
        y: torch.Tensor,
        model_comm_group: ProcessGroup,
        grid_dim: int,
        grid_shard_shapes: list,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple | None]:
        """Prepare tensors for smoothing.

        Args:
            y_pred_ens: torch.Tensor
                Ensemble predictions
            y: torch.Tensor
                Ground truth
            model_comm_group: ProcessGroup
                Model communication group

        Returns
        -------
            y_pred_ens_interp: torch.Tensor
                Predictions for interpolation
            y_interp: torch.Tensor
                Ground truth for interpolation
            shard_info: tuple
                Shard shapes for later gathering
        """
        batch_size, ensemble_size = y_pred_ens.shape[0], y_pred_ens.shape[1]
        y_pred_ens_interp = einops.rearrange(y_pred_ens, "b e g c -> (b e) g c")
        shard_shapes = apply_shard_shapes(y_pred_ens_interp, grid_dim, grid_shard_shapes)
        y_pred_ens_interp = shard_channels(y_pred_ens_interp, shard_shapes, model_comm_group)
        y_pred_ens_interp = einops.rearrange(
            y_pred_ens_interp,
            "(b e) g c -> b e g c",
            b=batch_size,
            e=ensemble_size,
        )

        shard_shapes_y = apply_shard_shapes(y, grid_dim, grid_shard_shapes)
        y_interp = shard_channels(y, shard_shapes_y, model_comm_group)

        return y_pred_ens_interp, y_interp, shard_shapes, shard_shapes_y

    def _apply_projector(self, batch: torch.Tensor, provider: ProjectionGraphProvider) -> torch.Tensor:
        """Apply sparse projector to a batch, handling multi-dimensional inputs."""
        input_shape = batch.shape
        batch = batch.reshape(-1, *input_shape[-2:])
        projection_matrix = provider.get_edges(device=batch.device)
        batch = self.projector(batch, projection_matrix)
        return batch.reshape(*input_shape[:-2] + batch.shape[-2:])

    def _smooth_for_loss(self, x: torch.Tensor, y: torch.Tensor, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply smoothing matrix to predictions and targets for loss computation."""
        if self.smoothing_matrices[i] is not None:
            x = self._apply_projector(x, self.smoothing_matrices[i])
            y = self._apply_projector(y, self.smoothing_matrices[i])
        return x, y

    def forward(
        self,
        y_pred_ens: torch.Tensor,
        y: torch.Tensor,
        squash: bool = True,
        grid_shard_slice: tuple | None = None,
        model_comm_group: ProcessGroup | None = None,
        model_comm_group_size: int | None = None,
        grid_dim: int | None = None,
        grid_shard_shapes: list | None = None,
        **_kwargs,
    ) -> list[torch.Tensor]:

        shard_shapes, shard_shapes_y = None, None
        if model_comm_group_size and model_comm_group_size > 1 and self.keep_batch_sharded:
            # go to full sequence dimension for smoothing
            y_pred_ens_for_smooth, y_for_smooth, shard_shapes, shard_shapes_y = self._prepare_for_smoothing(
                y_pred_ens,
                y,
                model_comm_group,
                grid_dim,
                grid_shard_shapes,
            )
        else:
            y_pred_ens_for_smooth = y_pred_ens
            y_for_smooth = y

        loss_inc = []
        y_preds_ens = []
        y_ens = []
        for i, provider in enumerate(self.smoothing_matrices):
            LOGGER.debug(
                "Loss: %s %s",
                i,
                provider.get_edges().shape if provider is not None else None,
            )

            # smooth the predictions and the truth for loss computation
            y_pred_ens_tmp, y_tmp = self._smooth_for_loss(y_pred_ens_for_smooth, y_for_smooth, i)

            if model_comm_group_size and model_comm_group_size > 1 and self.keep_batch_sharded:
                y_pred_ens_tmp = gather_channels(y_pred_ens_tmp, shard_shapes, model_comm_group)
                y_tmp = gather_channels(y_tmp, shard_shapes_y, model_comm_group)

            # save for next loss scale
            y_preds_ens.append(y_pred_ens_tmp)
            y_ens.append(y_tmp)

            if i > 0:  # assumption, resol 0 < 1 < 2 < ... < n
                y_pred_ens_tmp = y_pred_ens_tmp - y_preds_ens[i - 1]
                y_tmp = y_tmp - y_ens[i - 1]

            # compute the loss
            loss_inc.append(
                self.loss(
                    y_pred_ens_tmp,
                    y_tmp,
                    squash=squash,
                    grid_shard_slice=grid_shard_slice,
                    group=model_comm_group,
                ),
            )

        weighted_losses = [w * loss_val for w, loss_val in zip(self.weights, loss_inc, strict=True)]
        return torch.stack(weighted_losses)
