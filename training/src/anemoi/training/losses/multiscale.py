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

import einops
import torch
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.graphs.builders import _expand_smoother_config
from anemoi.graphs.builders import build_smoother_subgraph
from anemoi.graphs.projection_helpers import DEFAULT_DATASET_NAME
from anemoi.graphs.projection_helpers import DEFAULT_EDGE_WEIGHT_ATTRIBUTE
from anemoi.models.distributed.graph import gather_channels
from anemoi.models.distributed.graph import shard_channels
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.models.layers.sparse_projector import SparseProjector
from anemoi.training.losses.base import BaseLoss

LOGGER = logging.getLogger(__name__)


class MultiscaleLossWrapper(BaseLoss):
    """Apply the same base loss across progressively smoothed target fields."""

    name: str = "MultiscaleLossWrapper"
    needs_graph_data: bool = True

    def __init__(
        self,
        per_scale_loss: BaseLoss,
        weights: list[float],
        keep_batch_sharded: bool,
        multiscale_config: object | None = None,
        graph_data: HeteroData | None = None,
        data_node_name: str = DEFAULT_DATASET_NAME,
        autocast: bool = False,
        ignore_nans: bool = False,
        # Deprecated: pass loss_matrices_path / loss_matrices inside multiscale_config instead.
        loss_matrices_path: Path | str | None = None,
        loss_matrices: list[Path | str] | None = None,
    ) -> None:
        """Wrapper for multi-scale loss computation.

        Parameters
        ----------
        per_scale_loss : BaseLoss
            Loss to be used at each scale
        weights : list[float]
            Per-scale loss weights
        keep_batch_sharded : bool
            Whether the task should keep the batch grid-sharded during loss
            computation. When enabled, the task passes shard-layout metadata to
            this wrapper and multiscale smoothing follows the sharded path.
            If disabled, the loss is evaluated on replicated full-grid tensors
            on each model rank.
        multiscale_config : object | None
            Configuration for the smoothing matrices.  Accepts two forms:

            - **File mode** provide ``loss_matrices`` (list of filenames) and
              optionally ``loss_matrices_path`` (directory prefix)::

                multiscale_config:
                  loss_matrices_path: /path/to/dir
                  loss_matrices:
                    - filter_8x.npz   # coarsest
                    - filter_4x.npz
                    - null            # full resolution

            - **On-the-fly mode** provide a compact geometric-progression spec
              or an explicit ``smoothers`` mapping (passed to
              ``_expand_smoother_config``)::

                multiscale_config:
                  num_scales: 3
                  base_num_nearest_neighbours: 4
                  base_sigma: 0.1
                  scale_factor: 2

        graph_data : HeteroData | None
            Main graph; required for on-the-fly mode to copy data-node positions.
        data_node_name : str
            Node type in *graph_data* that holds the data-grid coordinates.
        autocast : bool
            Whether to use automatic mixed precision for the projections.
        ignore_nans : bool
            Passed to :class:`BaseLoss`; ignored by the wrapper itself.
        loss_matrices_path : Path | str | None
            Deprecated.  Pass inside *multiscale_config* instead.
        loss_matrices : list[Path | str] | None
            Deprecated.  Pass inside *multiscale_config* instead.
        """
        super().__init__(ignore_nans=ignore_nans)

        _has_matrices = loss_matrices and any(m is not None for m in loss_matrices)
        if _has_matrices or loss_matrices_path is not None:
            LOGGER.warning(
                "Passing 'loss_matrices' / 'loss_matrices_path' as top-level kwargs is deprecated. "
                "Move them inside 'multiscale_config' instead.",
            )
            cfg = dict(multiscale_config) if multiscale_config is not None else {}
            if _has_matrices:
                cfg.setdefault("loss_matrices", loss_matrices)
            if loss_matrices_path is not None:
                cfg.setdefault("loss_matrices_path", loss_matrices_path)
            multiscale_config = cfg

        self.smoothing_matrices = self._load_smoothing_matrices(
            multiscale_config,
            graph_data,
            data_node_name,
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

    @property
    def needs_shard_layout_info(self) -> bool:
        """Whether the wrapper needs shard-layout metadata from the task.

        This is tied to ``keep_batch_sharded`` because the wrapper only
        redistributes tensors across the model group when the batch remains
        grid-sharded during loss computation.
        """
        return self.keep_batch_sharded

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
        multiscale_config: object | None,
        graph_data: HeteroData | None,
        data_node_name: str,
    ) -> list[ProjectionGraphProvider | None]:
        """Load smoothing matrices for multi-scale loss computation.

        Dispatches to file mode when *multiscale_config* contains a
        ``loss_matrices`` key, otherwise to on-the-fly graph mode.
        """
        if multiscale_config is None:
            LOGGER.info("No multiscale_config specified, using single scale without smoothing")
            return [None]

        from omegaconf import OmegaConf

        cfg = (
            OmegaConf.to_container(multiscale_config, resolve=True)
            if OmegaConf.is_config(multiscale_config)
            else dict(multiscale_config)
        )

        if "loss_matrices" in cfg:
            return self._load_file_smoothing_matrices(
                cfg.get("loss_matrices_path"),
                cfg["loss_matrices"],
            )

        assert graph_data is not None, "graph_data must be provided for on-the-fly multiscale_config."
        return self._build_graph_smoothing_matrices(cfg, graph_data, data_node_name)

    def _build_graph_smoothing_matrices(
        self,
        multiscale_config: object,
        graph_data: HeteroData,
        data_node_name: str,
    ) -> list[ProjectionGraphProvider | None]:
        """Build one projection provider per smoother scale from config."""
        smoothers = _expand_smoother_config(multiscale_config)
        assert smoothers, "multiscale_config must define smoothers (explicit or via num_scales)."

        smoothing_matrices: list[ProjectionGraphProvider | None] = []
        edge_name = (data_node_name, "to", data_node_name)

        # Reverse order: coarsest scale first (highest smoothing)
        for smoother_name, smoother_cfg in reversed(list(smoothers.items())):
            subgraph = build_smoother_subgraph(graph_data, data_node_name, smoother_cfg)
            src_node_weight_attribute = (
                smoother_cfg.get("src_node_weight_attribute") if isinstance(smoother_cfg, dict) else None
            )
            row_normalize = bool(smoother_cfg.get("row_normalize", False)) if isinstance(smoother_cfg, dict) else False
            provider = ProjectionGraphProvider(
                graph=subgraph,
                edges_name=edge_name,
                edge_weight_attribute=DEFAULT_EDGE_WEIGHT_ATTRIBUTE,
                src_node_weight_attribute=src_node_weight_attribute,
                row_normalize=row_normalize,
            )
            smoothing_matrices.append(provider)
            LOGGER.info("Loss smoothing (graph, %s): %s", smoother_name, provider.get_edges().shape)

        smoothing_matrices.append(None)  # full-resolution scale — no smoothing
        return smoothing_matrices

    def _load_file_smoothing_matrices(
        self,
        loss_matrices_path: Path | str | None,
        loss_matrices: list[Path | str] | None,
    ) -> list[ProjectionGraphProvider | None]:
        """Create file-backed projection providers from serialized sparse matrices."""
        if not loss_matrices:
            LOGGER.info("No smoothing files specified, using single scale without smoothing")
            return [None]

        smoothing_matrices: list[ProjectionGraphProvider | None] = []
        for filename in loss_matrices:
            # Skip None, False, or the string "None"
            if filename is None or filename is False or filename == "None":
                smoothing_matrices.append(None)
                LOGGER.info("Loss smoothing: %s", None)
                continue

            file_path = Path(filename) if loss_matrices_path is None else Path(loss_matrices_path, filename)
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
        group: ProcessGroup | None,
        grid_dim: int,
        grid_shard_shapes: list,
    ) -> tuple[torch.Tensor, torch.Tensor, list | None, list | None]:
        """Prepare tensors for smoothing.

        Args:
            y_pred_ens: torch.Tensor
                Ensemble predictions
            y: torch.Tensor
                Ground truth
            group: ProcessGroup | None
                Model communication group

        Returns
        -------
            y_pred_ens_interp: torch.Tensor
                Predictions for interpolation
            y_interp: torch.Tensor
                Ground truth for interpolation
            shard_shapes: list | None
                Prediction shard shapes for later gathering
            shard_shapes_y: list | None
                Target shard shapes for later gathering
        """
        batch_size, out_times, ensemble_size = y_pred_ens.shape[0], y_pred_ens.shape[1], y_pred_ens.shape[2]
        y_pred_ens_interp = einops.rearrange(y_pred_ens, "b t e g c -> (b e) t g c")
        shard_shapes = apply_shard_shapes(y_pred_ens_interp, grid_dim, grid_shard_shapes)
        y_pred_ens_interp = shard_channels(y_pred_ens_interp, shard_shapes, group)
        y_pred_ens_interp = einops.rearrange(
            y_pred_ens_interp,
            "(b e) t g c -> b t e g c",
            b=batch_size,
            e=ensemble_size,
            t=out_times,
        )

        shard_shapes_y = apply_shard_shapes(y, grid_dim, grid_shard_shapes)
        y_interp = shard_channels(y, shard_shapes_y, group)

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
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        grid_dim: int | None = None,
        grid_shard_shapes: list | None = None,
        **kwargs,
    ) -> torch.Tensor:
        shard_shapes, shard_shapes_y = None, None
        is_model_sharded = grid_shard_shapes is not None
        if is_model_sharded:
            # go to full sequence dimension for smoothing
            y_pred_ens_for_smooth, y_for_smooth, shard_shapes, shard_shapes_y = self._prepare_for_smoothing(
                y_pred_ens,
                y,
                group,
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

            if is_model_sharded:
                y_pred_ens_tmp = gather_channels(y_pred_ens_tmp, shard_shapes, group)
                y_tmp = gather_channels(y_tmp, shard_shapes_y, group)

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
                    scaler_indices=scaler_indices,
                    without_scalers=without_scalers,
                    grid_shard_slice=grid_shard_slice,
                    group=group,
                    **kwargs,
                ),
            )

        weighted_losses = [w * loss_val for w, loss_val in zip(self.weights, loss_inc, strict=True)]
        return torch.stack(weighted_losses)
