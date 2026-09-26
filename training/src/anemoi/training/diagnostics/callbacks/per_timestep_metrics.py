# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Callback to log per-timestep validation metrics for temporal downscaling tasks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.index_space import IndexSpace

if TYPE_CHECKING:
    from anemoi.models.data import Batch
    from anemoi.models.data.sources import Source
    from anemoi.training.train.step_output import TrainingStepOutput

LOGGER = logging.getLogger(__name__)


class PerTimestepMetrics(Callback):
    """Log validation metrics broken down by output timestep.

    For tasks where the model predicts multiple
    output timesteps at once, this callback slices predictions and targets
    along the time dimension and logs per-timestep validation metrics.

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
        outputs: TrainingStepOutput | None,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        if batch_idx % self.every_n_batches != 0:
            return

        # validation_step returns a TrainingStepOutput whose predictions hold one
        # {dataset_name: Source} per task step, with ensemble members already gathered
        if outputs is None or not outputs.predictions:
            return

        with torch.no_grad():
            self._eval_per_timestep(pl_module, outputs.predictions, batch)

    def _eval_per_timestep(
        self,
        pl_module: pl.LightningModule,
        y_preds_list: list[dict[str, Source]],
        batch: Batch,
    ) -> None:
        """Compute metrics per timestep from the validation predictions, without another forward pass."""
        # Use the first (and typically only) task step's predictions, and the targets of that same step
        first_step_kwargs = next(iter(pl_module.task.steps("validation")))
        raw_y, _ = pl_module.task.get_targets(batch, data_indices=pl_module.data_indices, **first_step_kwargs)
        y_targets = pl_module.preprocess_targets(raw_y)
        y_preds = y_preds_list[0]

        for dataset_name, y_pred in y_preds.items():
            y = y_targets[dataset_name]
            # tabular sources count their time slots from the boundaries, gridded ones from the time axis
            n_timesteps = y.time_size

            # Gather the grid up front when any loss/metric does not support sharding, so non-sharding
            # metrics (e.g. spectral) get the full grid
            y_pred, y, grid_shard_slice = pl_module._prepare_tensors_for_loss(
                y_pred,
                y,
                dataset_name=dataset_name,
                validation_mode=True,
            )

            for t in range(n_timesteps):
                # Delegate to calculate_val_metrics which handles:
                # - post-processing (aligned to each view's index space)
                # - metric loop and metric ranges
                # - metric kwargs (scaler_indices, shard info, layouts)
                metrics = pl_module.calculate_val_metrics(
                    y_pred.select(time=slice(t, t + 1)),
                    y.select(time=slice(t, t + 1)),
                    grid_shard_slice=grid_shard_slice,
                    dataset_name=dataset_name,
                    pred_layout=IndexSpace.MODEL_OUTPUT,
                    target_layout=IndexSpace.DATA_FULL,
                    without_scalers=[TensorDim.TIME.value],
                )

                for metric_name, value in metrics.items():
                    pl_module.log(
                        f"val_{metric_name}/t_{t + 1}",
                        value,
                        on_epoch=True,
                        on_step=False,
                        prog_bar=False,
                        logger=pl_module.logger_enabled,
                        batch_size=batch.batch_size,
                        sync_dist=True,
                    )
