# (C) Copyright 2024 Anemoi contributors.
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
from scipy.sparse import load_npz
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import gather_channels
from anemoi.models.distributed.graph import shard_channels
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.truncation import interpolate_batch
from anemoi.models.truncation import make_truncation_matrix
from anemoi.training.losses.base import BaseLoss

LOGGER = logging.getLogger(__name__)


class MultiscaleLossWrapper(BaseLoss):

    name: str = "MultiscaleLossWrapper"

    def __init__(
        self,
        per_scale_loss: BaseLoss,
        weights: list[float],
        keep_batch_sharded: bool,
        truncation_path: Path | str | None = None,
        filenames: list[Path | str] | None = None,
    ) -> None:
        """Wrapper for multi-scale loss computation.

        Parameters
        ----------
        truncation_path : Path | str
            Path to the truncation matrices
        filenames : list[Path | str] | None
            Filenames of the truncation matrices
        weights : list[float]
            Per-scale loss weights
        keep_batch_sharded : bool
            Whether to keep the batch sharded during loss computation
        internal_loss : BaseLoss
            Loss to be used at each scale
        """
        super().__init__()

        self.truncation_matrices = self.load_loss_truncation_matrices(truncation_path, filenames)
        self.num_scales = len(self.truncation_matrices)
        assert (
            len(weights) == self.num_scales
        ), f"Number of weights ({len(weights)}) must match number of scales ({self.num_scales})"
        self.weights = weights
        self.loss = per_scale_loss
        self.scaler = self.loss.scaler
        self.keep_batch_sharded = keep_batch_sharded
        self.supports_sharding = True
        self.mloss = None

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

    def load_loss_truncation_matrices(
        self,
        truncation_path: Path | str,
        filenames: list[Path | str] | None,
    ) -> list[torch.Tensor | None]:

        # for loss decomposition
        truncation_matrices = []

        # Handle None, empty list, or falsy values - default to single scale with no truncation
        if not filenames:
            LOGGER.info("No truncation files specified, using single scale without truncation")
            return [None]

        for interp_data_loss in filenames:
            # Skip None, False, or the string "None"
            if interp_data_loss is None or interp_data_loss is False or interp_data_loss == "None":
                truncation_matrices.append(None)
                LOGGER.info("Loss truncation: %s", None)
            else:
                truncation_matrix = load_npz(Path(truncation_path, interp_data_loss))
                truncation_matrices.append(make_truncation_matrix(truncation_matrix))
                LOGGER.info("Loss truncation: %s %s", truncation_matrix.shape[0], truncation_matrix.shape[1])

        return truncation_matrices

    def _prepare_for_truncation(
        self,
        y_pred_ens: torch.Tensor,
        y: torch.Tensor,
        model_comm_group: ProcessGroup,
        grid_dim: int,
        grid_shard_shapes: list,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple | None]:
        """Prepare tensors for interpolation/smoothing.

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

    def _interp_for_loss(self, x: torch.Tensor, y: torch.Tensor, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.truncation_matrices[i] is not None:
            self.truncation_matrices[i] = self.truncation_matrices[i].to(x.device)
            x = interpolate_batch(x, self.truncation_matrices[i])
            y = interpolate_batch(y, self.truncation_matrices[i])
        return x, y

    def forward(
        self,
        y_pred_ens: torch.Tensor,
        y: torch.Tensor,
        squash: bool = True,  # noqa: ARG002
        grid_shard_slice: tuple | None = None,
        model_comm_group: ProcessGroup | None = None,
        model_comm_group_size: int | None = None,
        grid_dim: int | None = None,
        grid_shard_shapes: list | None = None,
        **_kwargs,
    ) -> list[torch.Tensor]:

        shard_shapes, shard_shapes_y = None, None
        if model_comm_group_size and model_comm_group_size > 1 and self.keep_batch_sharded:
            # go to full sequence dimension for interpolation / smoothing
            y_pred_ens_for_interp, y_for_interp, shard_shapes, shard_shapes_y = self._prepare_for_truncation(
                y_pred_ens,
                y,
                model_comm_group,
                grid_dim,
                grid_shard_shapes,
            )
        else:
            y_pred_ens_for_interp = y_pred_ens
            y_for_interp = y

        loss_inc = []
        y_preds_ens = []
        y_ens = []
        for i, trunc_matrix in enumerate(self.truncation_matrices):
            LOGGER.debug(
                "Loss: %s %s %s",
                i,
                trunc_matrix.shape if trunc_matrix is not None else None,
                trunc_matrix.device if trunc_matrix is not None else None,
            )

            # interpolate / smooth the predictions and the truth for loss computation
            y_pred_ens_tmp, y_tmp = self._interp_for_loss(y_pred_ens_for_interp, y_for_interp, i)

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
                    squash=True,
                    grid_shard_slice=grid_shard_slice,
                    group=model_comm_group,
                ),
            )

        weighted_losses = [w * loss_val for w, loss_val in zip(self.weights, loss_inc, strict=True)]
        return torch.stack(weighted_losses)
