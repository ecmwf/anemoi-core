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

from anemoi.models.distributed.graph import all_to_all_transpose
from anemoi.models.distributed.shapes import get_shard_sizes
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
        """
        super().__init__()

        self.smoothing_matrices = self._load_smoothing_matrices(loss_matrices_path, loss_matrices)
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
        loss_matrices_path: Path | str,
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
                provider = ProjectionGraphProvider(
                    file_path=Path(loss_matrices_path, filename),
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
        grid_shard_sizes: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
        """Prepare tensors for smoothing.

        Transitions from grid-sharded to channel-sharded layout via all-to-all
        so that smoothing (which needs the full grid) can run locally.

        Returns
        -------
            y_pred_ens_interp, y_interp, channel_shard_sizes
        """
        batch_size, out_times, ensemble_size = y_pred_ens.shape[0], y_pred_ens.shape[1], y_pred_ens.shape[2]
        y_pred_ens_interp = einops.rearrange(y_pred_ens, "b t e g c -> (b e) g (c t)")

        # grid-sharded -> channel-sharded: split along channels (dim_split=-1), concat along grid (dim_concat=-2)
        channel_shard_sizes = get_shard_sizes(y_pred_ens_interp, -1, model_comm_group)
        y_pred_ens_interp = all_to_all_transpose(
            y_pred_ens_interp,
            -1,
            channel_shard_sizes,
            -2,
            grid_shard_sizes,
            model_comm_group,
        )
        y_pred_ens_interp = einops.rearrange(
            y_pred_ens_interp,
            "(b e) g (c t) -> b t e g c",
            b=batch_size,
            e=ensemble_size,
            t=out_times,
        )

        y_interp = all_to_all_transpose(
            y,
            -1,
            channel_shard_sizes,
            -2,
            grid_shard_sizes,
            model_comm_group,
        )

        return y_pred_ens_interp, y_interp, channel_shard_sizes

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

    @property
    def needs_shard_layout_info(self) -> bool:
        """Whether the wrapper needs shard-layout metadata from the task."""
        return self.keep_batch_sharded

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
        grid_shard_sizes: list[int] | None = None,
        **kwargs,
    ) -> torch.Tensor:

        channel_shard_sizes = None
        is_model_sharded = grid_shard_sizes is not None
        if is_model_sharded and self.keep_batch_sharded:
            # go to full sequence dimension for smoothing
            y_pred_ens_for_smooth, y_for_smooth, channel_shard_sizes = self._prepare_for_smoothing(
                y_pred_ens,
                y,
                group,
                grid_shard_sizes,
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

            if is_model_sharded and self.keep_batch_sharded:
                # channel-sharded -> grid-sharded: reverse the all-to-all
                y_pred_ens_tmp = all_to_all_transpose(
                    y_pred_ens_tmp,
                    -2,
                    grid_shard_sizes,
                    -1,
                    channel_shard_sizes,
                    group,
                )
                y_tmp = all_to_all_transpose(
                    y_tmp,
                    -2,
                    grid_shard_sizes,
                    -1,
                    channel_shard_sizes,
                    group,
                )

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
