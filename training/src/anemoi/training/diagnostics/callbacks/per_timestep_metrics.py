# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Callback to log per-timestep validation metrics for temporal downscaling tasks."""

import logging

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.index_space import IndexSpace

LOGGER = logging.getLogger(__name__)


class PerTimestepMetrics(Callback):
    """Log validation metrics broken down by output timestep.

    For tasks where the model predicts multiple output timesteps at once,
    this callback slices predictions and targets along the time dimension
    and logs per-timestep validation metrics.

    It reuses the predictions already computed by the validation step (via
    ``outputs``) and delegates metric computation to
    ``calculate_val_metrics``, ensuring it stays in sync with the main
    validation logic.

    Parameters
    ----------
    every_n_batches : int
        Frequency of per-timestep evaluation (runs every N validation batches).
        Default is 1 (every batch).
    """

    def __init__(self, every_n_batches: int = 1) -> None:
        super().__init__()
        self.every_n_batches = every_n_batches

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,  # noqa: ARG002
        pl_module: pl.LightningModule,
        outputs: tuple,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        if batch_idx % self.every_n_batches != 0:
            return

        # validation_step returns a TrainingStepOutput dataclass
        # y_preds is a list of {dataset_name: tensor}, one entry per task step
        if outputs is None:
            return

        y_preds_list = outputs.predictions
        if not y_preds_list:
            return

        batch_size = next(iter(batch.values())).shape[0]

        with torch.no_grad():
            self._eval_per_timestep(pl_module, y_preds_list, batch, batch_size)

    def _eval_per_timestep(
        self,
        pl_module: pl.LightningModule,
        y_preds_list: list[dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
        batch_size: int,
    ) -> None:
        """Compute metrics per timestep using validation outputs."""
        # Get targets from batch (cheap — just indexing)
        y_targets = pl_module.task.get_targets(batch)
        if hasattr(pl_module, "_collapse_ens_dim"):
            y_targets = pl_module._collapse_ens_dim(y_targets)

        # Use the first (and typically only) task step's predictions
        y_preds = y_preds_list[0]

        for dataset_name, y_pred in y_preds.items():
            y = y_targets[dataset_name]
            n_timesteps = y.shape[TensorDim.TIME]
            grid_shard_slice = pl_module.grid_shard_slice.get(dataset_name)

            for t in range(n_timesteps):
                # Slice single timestep, keeping the time dimension
                pred_t = y_pred[:, t : t + 1]
                target_t = y[:, t : t + 1]

                # Delegate to calculate_val_metrics which handles:
                # - post-processing
                # - metric loop and metric ranges
                # - metric kwargs (scaler_indices, shard info, layouts)
                metrics = pl_module.calculate_val_metrics(
                    pred_t,
                    target_t,
                    grid_shard_slice=grid_shard_slice,
                    dataset_name=dataset_name,
                    pred_layout=IndexSpace.MODEL_OUTPUT,
                    target_layout=IndexSpace.DATA_FULL,
                )

                for metric_name, value in metrics.items():
                    step_name = f"val_{metric_name}/t_{t + 1}"

                    pl_module.log(
                        step_name,
                        value,
                        on_epoch=True,
                        on_step=False,
                        prog_bar=False,
                        logger=pl_module.logger_enabled,
                        batch_size=batch_size,
                        sync_dist=True,
                    )
