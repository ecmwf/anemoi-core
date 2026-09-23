# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import Iterator
from pathlib import Path

import einops
import torch
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.graphs.builders import _expand_smoother_config
from anemoi.graphs.builders import build_smoother_subgraph
from anemoi.graphs.projection_helpers import DEFAULT_DATASET_NAME
from anemoi.graphs.projection_helpers import DEFAULT_EDGE_WEIGHT_ATTRIBUTE
from anemoi.models.distributed.graph import all_to_all_transpose
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.models.layers.sparse_projector import SparseProjector
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import BaseLossWrapper
from anemoi.training.losses.spectral_scales import SpectralScales
from anemoi.training.losses.spectral_scales import build_spectral_scales

LOGGER = logging.getLogger(__name__)


class MultiscaleLossWrapper(BaseLossWrapper):

    name: str = "MultiscaleLossWrapper"
    needs_graph_data: bool = True
    needs_data_node_name: bool = True

    def __init__(
        self,
        per_scale_loss: BaseLoss,
        weights: list[float],
        multiscale_config: object | None = None,
        graph_data: HeteroData | None = None,
        data_node_name: str = DEFAULT_DATASET_NAME,
        autocast: bool = False,
        sparse_projector_num_chunks: int = 1,
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
        multiscale_config : object | None
            Configuration for the loss scales.  Accepts three forms:

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

            - **Spectral mode** provide a ``transform``, its grid and strictly
              increasing ``cutoffs``. Each scale keeps what lies at or below its
              cutoff; a full-resolution scale is added last. Grid points are
              ordered as for the spectral losses: ring by ring for the spherical
              harmonic transforms, ``(y x)`` for the rectangular ones.

              - ``octahedral_sht`` / ``regular_sht`` with ``nlat``, or
                ``reduced_sht`` with ``grid`` (e.g. ``n320``): cutoffs are
                integer truncations::

                    multiscale_config:
                      transform: octahedral_sht
                      nlat: 640
                      cutoffs: [79, 159, 319]   # four scales with the full resolution

              - ``dct2d`` or ``fft2d`` with ``x_dim`` and ``y_dim``, for
                limited-area grids: cutoffs are frequencies in cycles per grid
                spacing (0.125 keeps wavelengths of 8 grid spacings and longer).
                ``dct2d`` suits non-periodic domains, ``fft2d`` treats the domain
                as periodic::

                    multiscale_config:
                      transform: dct2d
                      x_dim: 1000
                      y_dim: 800
                      cutoffs: [0.03125, 0.0625, 0.125]

        graph_data : HeteroData | None
            Main graph; required for on-the-fly mode to copy data-node positions.
        data_node_name : str
            Node type in *graph_data* that holds the data-grid coordinates.
        autocast : bool
            Whether to use automatic mixed precision for the smoothing matrices.
            Spectral scales always run in float32.
        sparse_projector_num_chunks : int
            Default number of chunks for smoothing with matrices.
            ``1`` means processing all fields in one go.
        ignore_nans : bool
            Passed to :class:`BaseLoss`; ignored by the wrapper itself.
        loss_matrices_path : Path | str | None
            Deprecated.  Pass inside *multiscale_config* instead.
        loss_matrices : list[Path | str] | None
            Deprecated.  Pass inside *multiscale_config* instead.
        """
        super().__init__(loss=per_scale_loss, ignore_nans=ignore_nans)

        _has_matrices = bool(loss_matrices)  # [None] still signals file mode (identity scale)
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

        self.spectral_scales: SpectralScales | None = None
        self.smoothers = self._load_smoothers(
            multiscale_config,
            graph_data,
            data_node_name,
        )
        self.num_scales = len(self.smoothers)
        assert (
            len(weights) == self.num_scales
        ), f"Number of weights ({len(weights)}) must match number of scales ({self.num_scales})"
        self.weights = weights
        self.supports_sharding = True
        self.mloss = None
        self.projector = SparseProjector(autocast=autocast)
        self.sparse_projector_num_chunks = sparse_projector_num_chunks

    @property
    def needs_shard_layout_info(self) -> bool:
        return True

    def iter_leaf_losses(self) -> Iterator["BaseLoss"]:
        """MultiscaleLossWrapper is a leaf: it performs substantive computation."""
        yield self

    def _load_smoothers(
        self,
        multiscale_config: object | None,
        graph_data: HeteroData | None,
        data_node_name: str,
    ) -> list[ProjectionGraphProvider | float | None]:
        """Build one smoother per loss scale, coarsest first; None means full resolution.

        Dispatches to spectral mode when *multiscale_config* contains a
        ``transform`` key, to file mode when it contains a ``loss_matrices``
        key, otherwise to on-the-fly graph mode.
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

        if "transform" in cfg:
            return self._build_spectral_smoothers(cfg)

        if "loss_matrices" in cfg:
            from anemoi.training.schemas.training import MultiscaleConfigOnTheFlySchema

            onthefly_keys = set(MultiscaleConfigOnTheFlySchema.model_fields)
            if cfg.keys() & onthefly_keys:
                msg = (
                    "multiscale_config mixes file-based ('loss_matrices') and on-the-fly "
                    f"keys ({cfg.keys() & onthefly_keys}). Use one mode only."
                )
                raise ValueError(msg)
            return self._load_file_smoothing_matrices(
                cfg.get("loss_matrices_path"),
                cfg["loss_matrices"],
            )

        assert graph_data is not None, "graph_data must be provided for on-the-fly multiscale_config."
        return self._build_graph_smoothing_matrices(cfg, graph_data, data_node_name)

    def _build_spectral_smoothers(self, multiscale_config: dict) -> list[float | None]:
        """Set up the shared spectral transforms and return the cutoffs, coarsest first."""
        grid_kwargs = dict(multiscale_config)
        transform = grid_kwargs.pop("transform")
        cutoffs = list(grid_kwargs.pop("cutoffs"))
        if not cutoffs or cutoffs != sorted(set(cutoffs)):
            msg = f"multiscale_config 'cutoffs' must be strictly increasing, got {cutoffs}."
            raise ValueError(msg)

        self.spectral_scales = build_spectral_scales(transform, cutoffs, **grid_kwargs)
        LOGGER.info("Loss smoothing (%s): cutoffs %s", transform, cutoffs)
        return [*cutoffs, None]  # full-resolution scale last — no smoothing

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
            LOGGER.info(
                "Loss smoothing (graph, %s): %s",
                smoother_name,
                provider.get_edges().shape,
            )

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
        grid_shard_sizes: ShardSizes,
    ) -> tuple[torch.Tensor, torch.Tensor, list, list]:
        """Prepare tensors for smoothing.

        Transitions from grid-sharded to channel-sharded layout via all-to-all
        so that smoothing (which needs the full grid) can run locally.

        Returns
        -------
            y_pred_ens_interp, y_interp, channel_shard_sizes_pred, channel_shard_sizes_y
        """
        batch_size, out_times, ensemble_size = (
            y_pred_ens.shape[0],
            y_pred_ens.shape[1],
            y_pred_ens.shape[2],
        )
        y_pred_ens_interp = einops.rearrange(y_pred_ens, "b t e g c -> (b e) t g c")

        # grid-sharded -> channel-sharded: split along channels (dim_split=-1), concat along grid (dim_concat=-2)
        channel_shard_sizes_pred = get_shard_sizes(y_pred_ens_interp, -1, group)
        y_pred_ens_interp = all_to_all_transpose(
            y_pred_ens_interp,
            -1,
            channel_shard_sizes_pred,
            -2,
            grid_shard_sizes,
            group,
        )
        y_pred_ens_interp = einops.rearrange(
            y_pred_ens_interp,
            "(b e) t g c -> b t e g c",
            b=batch_size,
            e=ensemble_size,
            t=out_times,
        )

        channel_shard_sizes_y = get_shard_sizes(y, -1, group)
        y_interp = all_to_all_transpose(
            y,
            -1,
            channel_shard_sizes_y,
            -2,
            grid_shard_sizes,
            group,
        )

        return (
            y_pred_ens_interp,
            y_interp,
            channel_shard_sizes_pred,
            channel_shard_sizes_y,
        )

    def _smooth_for_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        smoother: ProjectionGraphProvider | float | None,
        x_coeffs: torch.Tensor | None = None,
        y_coeffs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Smooth predictions and targets to one loss scale.

        In spectral mode a scale is given by its cutoff and is built from the
        coefficients ``x_coeffs`` and ``y_coeffs`` of the shared analysis.
        """
        if smoother is None:
            return x, y
        if self.spectral_scales is not None:
            return (
                self.spectral_scales.synthesise(x_coeffs, smoother),
                self.spectral_scales.synthesise(y_coeffs, smoother),
            )
        projection_matrix = smoother.get_edges(device=x.device)
        x = self.projector(
            x,
            projection_matrix,
            num_chunks=self.sparse_projector_num_chunks,
        )
        y = self.projector(
            y,
            projection_matrix,
            num_chunks=self.sparse_projector_num_chunks,
        )
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
        grid_shard_sizes: ShardSizes = None,
        grid_dim: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        channel_shard_sizes_pred = None
        channel_shard_sizes_y = None
        is_model_sharded = grid_shard_sizes is not None
        if is_model_sharded:
            # go to full sequence dimension for smoothing
            (
                y_pred_ens_for_smooth,
                y_for_smooth,
                channel_shard_sizes_pred,
                channel_shard_sizes_y,
            ) = self._prepare_for_smoothing(
                y_pred_ens,
                y,
                group,
                grid_shard_sizes,
            )
        else:
            y_pred_ens_for_smooth = y_pred_ens
            y_for_smooth = y

        pred_coeffs = y_coeffs = None
        if self.spectral_scales is not None:
            # One forward transform per tensor serves every spectral scale.
            pred_coeffs = self.spectral_scales.analyse(y_pred_ens_for_smooth)
            y_coeffs = self.spectral_scales.analyse(y_for_smooth)

        weighted_losses = []
        prev_y_pred_ens = None
        prev_y = None
        for i, (weight, smoother) in enumerate(zip(self.weights, self.smoothers, strict=True)):
            if LOGGER.isEnabledFor(logging.DEBUG):
                # A spectral scale shows its cutoff; None is the full resolution.
                scale = smoother.projection_matrix.shape if isinstance(smoother, ProjectionGraphProvider) else smoother
                LOGGER.debug("Loss: %s %s", i, scale)

            # smooth the predictions and the truth for loss computation
            y_pred_ens_tmp, y_tmp = self._smooth_for_loss(
                y_pred_ens_for_smooth,
                y_for_smooth,
                smoother,
                pred_coeffs,
                y_coeffs,
            )

            if is_model_sharded:
                # channel-sharded -> grid-sharded: reverse the all-to-all
                y_pred_ens_tmp = all_to_all_transpose(
                    y_pred_ens_tmp,
                    -2,
                    grid_shard_sizes,
                    -1,
                    channel_shard_sizes_pred,
                    group,
                )
                y_tmp = all_to_all_transpose(
                    y_tmp,
                    -2,
                    grid_shard_sizes,
                    -1,
                    channel_shard_sizes_y,
                    group,
                )

            current_y_pred_ens = y_pred_ens_tmp
            current_y = y_tmp

            if prev_y_pred_ens is not None:  # assumption, resol 0 < 1 < 2 < ... < n
                y_pred_ens_tmp = y_pred_ens_tmp - prev_y_pred_ens
                y_tmp = y_tmp - prev_y

            prev_y_pred_ens = current_y_pred_ens
            prev_y = current_y

            # sharding kwargs - only pass if the loss needs them
            sharding_kwargs = (
                {"grid_shard_sizes": grid_shard_sizes, "grid_dim": grid_dim}
                if self.loss.needs_shard_layout_info
                else {}
            )
            # compute the loss
            weighted_losses.append(
                weight
                * self.loss(
                    y_pred_ens_tmp,
                    y_tmp,
                    squash=squash,
                    scaler_indices=scaler_indices,
                    without_scalers=without_scalers,
                    grid_shard_slice=grid_shard_slice,
                    group=group,
                    **sharding_kwargs,
                    **kwargs,
                ),
            )

        # The scale dimension is internal to this wrapper. Return the same
        # scalar or per-variable shape as the wrapped loss.
        return torch.stack(weighted_losses).sum(dim=0)
