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
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import gather_channels
from anemoi.models.distributed.graph import shard_channels
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.truncation import interpolate_batch
from anemoi.models.truncation import make_truncation_matrix
from anemoi.training.losses.base import BaseLoss

LOGGER = logging.getLogger(__name__)


class MultiscaleLossWrapper(nn.Module):

    name: str = "MultiscaleLossWrapper"

    def __init__(
        self,
        truncation_path: Path | str,
        filenames: Path | str,
        loss: BaseLoss,
        keep_batch_sharded: bool,
    ) -> None:
        """Wrapper for multi-scale loss computation.

        Parameters
        ----------
        truncation_path : Path | str
            Path to the truncation matrices
        filenames : Path | str
            Filenames of the truncation matrices
        loss : BaseLoss
            Loss to be used at each scale
        """
        super().__init__()

        self.truncation_matrices = self.load_loss_truncation_matrices(truncation_path, filenames)
        self.num_scales = len(self.truncation_matrices)
        self.loss = loss
        self.scaler = self.loss.scaler
        self.keep_batch_sharded = keep_batch_sharded

    def load_loss_truncation_matrices(
        self,
        truncation_path: Path | str,
        filenames: Path | str,
    ) -> list[torch.Tensor | None]:

        # for loss decomposition
        truncation_matrices = []
        for interp_data_loss in filenames:
            if interp_data_loss:
                truncation_matrix = load_npz(Path(truncation_path, interp_data_loss))
                truncation_matrices.append(make_truncation_matrix(truncation_matrix))
                LOGGER.info("Loss truncation: %s %s", truncation_matrix.shape[0], truncation_matrix.shape[1])
            else:
                truncation_matrices.append(None)
                LOGGER.info("Loss truncation: %s", None)

        return truncation_matrices

    def _prepare_for_truncation(
        self,
        y_pred_ens: torch.Tensor,
        y: torch.Tensor,
        model_comm_group: ProcessGroup,
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
        shard_shapes = apply_shard_shapes(y_pred_ens_interp, self.grid_dim, self.grid_shard_shapes)
        y_pred_ens_interp = shard_channels(y_pred_ens_interp, shard_shapes, model_comm_group)
        y_pred_ens_interp = einops.rearrange(
            y_pred_ens_interp,
            "(b e) g c -> b e g c",
            b=batch_size,
            e=ensemble_size,
        )

        shard_shapes_y = apply_shard_shapes(y, self.grid_dim, self.grid_shard_shapes)
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
        squash: bool,  # noqa: ARG002
        grid_shard_slice: tuple | None,
        model_comm_group: ProcessGroup,
    ) -> list[torch.Tensor]:

        shard_shapes, shard_shapes_y = None, None
        if self.keep_batch_sharded and torch.distributed.get_world_size() > 1:
            # go to full sequence dimension for interpolation / smoothing
            y_pred_ens_for_interp, y_for_interp, shard_shapes, shard_shapes_y = self._prepare_for_truncation(
                y_pred_ens,
                y,
                model_comm_group,
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

            if self.keep_batch_sharded and torch.distributed.get_world_size() > 1:
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

        loss = torch.stack(loss_inc).sum()
        mloss = [x.detach() for x in loss_inc]
        return loss, mloss
