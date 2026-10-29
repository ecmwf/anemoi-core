# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import functools
import logging
import math
import weakref
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import einops
import torch
import torch.utils.checkpoint
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.graphs.projection_helpers import DEFAULT_DATASET_NAME
from anemoi.models.distributed.graph import all_to_all_transpose
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.gaussian_smoothing import CompiledGaussianSmoothing
from anemoi.models.layers.gaussian_smoothing import gaussian_smooth
from anemoi.models.layers.graph_provider import NeighbourListProvider
from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.models.layers.projection_backend import ProjectionBackend
from anemoi.models.layers.sparse_projector import SparseProjector
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import BaseLossWrapper

if TYPE_CHECKING:
    from anemoi.training.schemas.training import MultiscaleConfigNeighbourListSchema

LOGGER = logging.getLogger(__name__)

# Neighbour lists by (graph, node set, K), so all multiscale losses on a grid share one list.
# Entries are dropped when their graph is freed.
_NEIGHBOUR_LISTS: dict[tuple[int, str, int], NeighbourListProvider] = {}


def shared_neighbour_list(graph_data: HeteroData, data_node_name: str, num_neighbours: int) -> NeighbourListProvider:
    """Return the shared neighbour list of a node set, building it on first use."""
    key = (id(graph_data), data_node_name, num_neighbours)
    if key not in _NEIGHBOUR_LISTS:
        _NEIGHBOUR_LISTS[key] = NeighbourListProvider(graph_data, data_node_name, num_neighbours)
        weakref.finalize(graph_data, _NEIGHBOUR_LISTS.pop, key, None)
    return _NEIGHBOUR_LISTS[key]


class NeighbourSmoothingScale:
    """One scale: Gaussian smoothing of width ``sigma`` on a shared neighbour list."""

    def __init__(self, neighbour_list: NeighbourListProvider, sigma: float, backend: ProjectionBackend) -> None:
        self.neighbour_list = neighbour_list
        self.sigma = sigma
        self.backend = backend
        self.compiled: CompiledGaussianSmoothing | None = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        neighbours, unit_vectors = self.neighbour_list.get_neighbours(device=x.device)
        return gaussian_smooth(x, neighbours, unit_vectors, self.sigma, backend=self.backend, compiled=self.compiled)

    def __repr__(self) -> str:
        return (
            f"NeighbourSmoothingScale(sigma={self.sigma}, neighbours={self.neighbour_list.num_neighbours}, "
            f"backend={self.backend!r}, compiled={self.compiled is not None})"
        )


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
        sparse_projector_backend: ProjectionBackend = "triton",
        ignore_nans: bool = False,
        checkpoint_scales: bool = False,
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
            Configuration of the smoothing.  Accepts two forms:

            - **File mode** provide ``loss_matrices`` (list of filenames) and
              optionally ``loss_matrices_path`` (directory prefix)::

                multiscale_config:
                  loss_matrices_path: /path/to/dir
                  loss_matrices:
                    - filter_8x.npz   # coarsest
                    - filter_4x.npz
                    - null            # full resolution

            - **Neighbour-list mode** provide the number of neighbours and one
              Gaussian width per scale, widest first. All scales share one
              neighbour list::

                multiscale_config:
                  num_nearest_neighbours: 128
                  sigmas: [0.0157, 0.0078, 0.0039]

        graph_data : HeteroData | None
            Main graph; needed in neighbour-list mode for the node positions.
        data_node_name : str
            Node type in *graph_data* that holds the data-grid coordinates.
        autocast : bool
            Whether to use automatic mixed precision for the projections.
        sparse_projector_num_chunks : int
            Default number of chunks for multiscale smoothing.
            ``1`` means processing all fields in one go.
        sparse_projector_backend : str
            ``"triton"`` (default, GPU) or ``"torch"`` (any device), in either mode.
        ignore_nans : bool
            Passed to :class:`BaseLoss`; ignored by the wrapper itself.
        checkpoint_scales : bool
            Recompute each scale in the backward pass instead of keeping it. The memory no longer
            grows with the number of scales, but every smoothing runs four times.
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

        self.smoothers = self._build_smoothers(
            multiscale_config,
            graph_data,
            data_node_name,
            sparse_projector_backend,
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
        self.checkpoint_scales = checkpoint_scales

    @property
    def needs_shard_layout_info(self) -> bool:
        return True

    def iter_leaf_losses(self) -> Iterator["BaseLoss"]:
        """MultiscaleLossWrapper is a leaf: it performs substantive computation."""
        yield self

    def compile_for_training(self, **options) -> None:
        """Compile the torch neighbour-list smoothing; hook for ``model.compile``.

        Does nothing for the Triton backend or file mode. The per-scale loss needs its own entry.
        """
        scales = [s for s in self.smoothers if isinstance(s, NeighbourSmoothingScale) and s.backend == "torch"]
        if not scales:
            LOGGER.info("MultiscaleLossWrapper: nothing to compile (no torch neighbour-list scales).")
            return
        compiled = CompiledGaussianSmoothing(**options)
        for scale in scales:
            scale.compiled = compiled

    def _build_smoothers(
        self,
        multiscale_config: object | None,
        graph_data: HeteroData | None,
        data_node_name: str,
        sparse_projector_backend: ProjectionBackend,
    ) -> list[ProjectionGraphProvider | NeighbourSmoothingScale | None]:
        """Build the smoother of each scale: neighbour-list mode for ``sigmas``, file mode for ``loss_matrices``."""
        if multiscale_config is None:
            LOGGER.info("No multiscale_config specified, using single scale without smoothing")
            return [None]

        from omegaconf import OmegaConf

        cfg = (
            OmegaConf.to_container(multiscale_config, resolve=True)
            if OmegaConf.is_config(multiscale_config)
            else dict(multiscale_config)
        )

        if "sigmas" in cfg:
            from anemoi.training.schemas.training import MultiscaleConfigNeighbourListSchema

            assert graph_data is not None, "graph_data must be provided for a neighbour-list multiscale_config."
            return self._build_neighbour_smoothing_scales(
                MultiscaleConfigNeighbourListSchema(**cfg),
                graph_data,
                data_node_name,
                sparse_projector_backend,
            )

        if "loss_matrices" in cfg:
            from anemoi.training.schemas.training import MultiscaleConfigDiskSchema

            other_keys = cfg.keys() - set(MultiscaleConfigDiskSchema.model_fields)
            if other_keys:
                msg = f"multiscale_config mixes file keys ('loss_matrices') with other keys ({other_keys})."
                raise ValueError(msg)
            return self._load_file_smoothing_matrices(
                cfg.get("loss_matrices_path"),
                cfg["loss_matrices"],
                sparse_projector_backend,
            )

        msg = f"multiscale_config needs 'sigmas' (neighbour-list mode) or 'loss_matrices' (file mode), got {set(cfg)}."
        raise ValueError(msg)

    def _build_neighbour_smoothing_scales(
        self,
        multiscale_config: "MultiscaleConfigNeighbourListSchema",
        graph_data: HeteroData,
        data_node_name: str,
        backend: ProjectionBackend,
    ) -> list[NeighbourSmoothingScale | None]:
        """One scale per sigma on a shared neighbour list, then the unsmoothed field."""
        neighbour_list = shared_neighbour_list(graph_data, data_node_name, multiscale_config.num_nearest_neighbours)
        scales: list[NeighbourSmoothingScale | None] = [
            NeighbourSmoothingScale(neighbour_list, sigma, backend) for sigma in multiscale_config.sigmas
        ]
        for scale in scales:
            LOGGER.info("Loss smoothing (neighbour list): %s", scale)
        reach, widest = neighbour_list.typical_reach(), multiscale_config.sigmas[0]
        LOGGER.info(
            "Loss smoothing: neighbours reach %.4f rad (median), %.1f x the widest sigma.",
            reach,
            reach / widest,
        )
        if reach < 2 * widest:
            LOGGER.warning(
                "The %d nearest neighbours reach only %.1f x the widest sigma (%.4f rad); its Gaussian is cut "
                "off at %.0f%% of the peak. Use more neighbours or a smaller widest sigma.",
                multiscale_config.num_nearest_neighbours,
                reach / widest,
                widest,
                100 * math.exp(-0.5 * (reach / widest) ** 2),
            )
        scales.append(None)  # the unsmoothed field is the last scale
        return scales

    def _load_file_smoothing_matrices(
        self,
        loss_matrices_path: Path | str | None,
        loss_matrices: list[Path | str] | None,
        backend: ProjectionBackend,
    ) -> list[ProjectionGraphProvider | None]:
        """Create file-backed projection providers from serialized sparse matrices."""
        if not loss_matrices:
            LOGGER.info("No smoothing files specified, using single scale without smoothing")
            return [None]

        smoothers: list[ProjectionGraphProvider | None] = []
        for filename in loss_matrices:
            # Skip None, False, or the string "None"
            if filename is None or filename is False or filename == "None":
                smoothers.append(None)
                LOGGER.info("Loss smoothing: %s", None)
                continue

            file_path = Path(filename) if loss_matrices_path is None else Path(loss_matrices_path, filename)
            provider = ProjectionGraphProvider(
                file_path=file_path,
                row_normalize=False,
                backend=backend,
            )
            smoothers.append(provider)
            LOGGER.info("Loss smoothing: %s, %s backend", provider.shape, backend)

        return smoothers

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

    def _smooth_for_loss(self, x: torch.Tensor, y: torch.Tensor, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Smooth predictions and targets at scale ``i`` for the loss."""
        if isinstance(self.smoothers[i], NeighbourSmoothingScale):
            return self.smoothers[i](x), self.smoothers[i](y)
        if self.smoothers[i] is not None:
            x = self.projector(
                x,
                self.smoothers[i],
                num_chunks=self.sparse_projector_num_chunks,
            )
            y = self.projector(
                y,
                self.smoothers[i],
                num_chunks=self.sparse_projector_num_chunks,
            )
        return x, y

    def _scale_fields(
        self,
        y_pred_ens_for_smooth: torch.Tensor,
        y_for_smooth: torch.Tensor,
        i: int,
        layout: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Smooth at scale ``i`` and return in the layout the per-scale loss expects."""
        y_pred_ens_tmp, y_tmp = self._smooth_for_loss(y_pred_ens_for_smooth, y_for_smooth, i)
        if layout["is_model_sharded"]:
            # channel-sharded -> grid-sharded: reverse the all-to-all
            y_pred_ens_tmp = all_to_all_transpose(
                y_pred_ens_tmp,
                -2,
                layout["grid_shard_sizes"],
                -1,
                layout["channel_shard_sizes_pred"],
                layout["group"],
            )
            y_tmp = all_to_all_transpose(
                y_tmp,
                -2,
                layout["grid_shard_sizes"],
                -1,
                layout["channel_shard_sizes_y"],
                layout["group"],
            )
        return y_pred_ens_tmp, y_tmp

    def _scale_loss(
        self,
        i: int,
        layout: dict,
        score: functools.partial,
        y_pred_ens_for_smooth: torch.Tensor,
        y_for_smooth: torch.Tensor,
    ) -> torch.Tensor:
        """Loss of scale ``i`` computed from the unsmoothed fields, for checkpointing.

        Scale ``i`` is the field smoothed at its own width minus the field at the next wider one.
        """
        y_pred_ens_tmp, y_tmp = self._scale_fields(y_pred_ens_for_smooth, y_for_smooth, i, layout)
        if i > 0:
            wider_y_pred_ens, wider_y = self._scale_fields(y_pred_ens_for_smooth, y_for_smooth, i - 1, layout)
            y_pred_ens_tmp = y_pred_ens_tmp - wider_y_pred_ens
            y_tmp = y_tmp - wider_y
        return score(y_pred_ens_tmp, y_tmp)

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

        layout = {
            "is_model_sharded": is_model_sharded,
            "group": group,
            "grid_shard_sizes": grid_shard_sizes,
            "channel_shard_sizes_pred": channel_shard_sizes_pred,
            "channel_shard_sizes_y": channel_shard_sizes_y,
        }
        # sharding kwargs - only pass if the loss needs them
        sharding_kwargs = (
            {"grid_shard_sizes": grid_shard_sizes, "grid_dim": grid_dim} if self.loss.needs_shard_layout_info else {}
        )
        score = functools.partial(
            self.loss,
            squash=squash,
            scaler_indices=scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
            group=group,
            **sharding_kwargs,
            **kwargs,
        )
        checkpointed = self.checkpoint_scales and torch.is_grad_enabled()

        weighted_losses = []
        prev_y_pred_ens = None
        prev_y = None
        for i, (weight, provider) in enumerate(zip(self.weights, self.smoothers, strict=True)):
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(
                    "Loss: %s %s",
                    i,
                    provider.shape if isinstance(provider, ProjectionGraphProvider) else provider,
                )

            if checkpointed:
                scale_loss = torch.utils.checkpoint.checkpoint(
                    functools.partial(self._scale_loss, i, layout, score),
                    y_pred_ens_for_smooth,
                    y_for_smooth,
                    use_reentrant=False,
                )
            else:
                # smooth the predictions and the truth for loss computation
                current_y_pred_ens, current_y = self._scale_fields(y_pred_ens_for_smooth, y_for_smooth, i, layout)
                y_pred_ens_tmp, y_tmp = current_y_pred_ens, current_y
                if prev_y_pred_ens is not None:  # assumption, resol 0 < 1 < 2 < ... < n
                    y_pred_ens_tmp = y_pred_ens_tmp - prev_y_pred_ens
                    y_tmp = y_tmp - prev_y
                prev_y_pred_ens = current_y_pred_ens
                prev_y = current_y
                scale_loss = score(y_pred_ens_tmp, y_tmp)

            weighted_losses.append(weight * scale_loss)

        # The scale dimension is internal to this wrapper. Return the same
        # scalar or per-variable shape as the wrapped loss.
        return torch.stack(weighted_losses).sum(dim=0)
