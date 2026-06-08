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
from contextlib import nullcontext

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from anemoi.training.losses.base import BaseLoss
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.index_space import IndexSpace

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
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list,  # noqa: ARG002
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        if batch_idx % self.every_n_batches != 0:
            return

        precision_mapping = {
            "16-mixed": torch.float16,
            "bf16-mixed": torch.bfloat16,
        }
        prec = trainer.precision
        dtype = precision_mapping.get(prec)

        context = (
            torch.autocast(device_type=next(iter(batch.values())).device.type, dtype=dtype)
            if dtype is not None
            else nullcontext()
        )

        with context, torch.no_grad():
            self._eval_per_timestep(pl_module, batch)

    def _eval_per_timestep(self, pl_module: pl.LightningModule, batch: dict[str, torch.Tensor]) -> None:
        """Run model and compute metrics per timestep."""
        # Get inputs and targets via the task
        x = pl_module.task.get_inputs(batch, data_indices=pl_module.data_indices)
        x = pl_module._expand_ens_dim(x) if hasattr(pl_module, "_expand_ens_dim") else x

        # Run model forward
        y_pred = pl_module(x)

        # Get targets
        y = pl_module.task.get_targets(batch)

        batch_size = next(iter(batch.values())).shape[0]

        # For each dataset, compute per-timestep metrics
        for dataset_name in y_pred:
            pred = y_pred[dataset_name]  # (bs, time, ens, grid, var)
            target = y[dataset_name]  # (bs, time, ens, grid, var)

            n_timesteps = target.shape[TensorDim.TIME]

            # Gather ensemble members across the ensemble comm group
            if hasattr(pl_module, "ens_comm_subgroup") and pl_module.ens_comm_subgroup is not None:
                from anemoi.models.distributed.graph import gather_tensor

                pred = gather_tensor(
                    pred.clone(),
                    dim=TensorDim.ENSEMBLE_DIM,
                    sizes=[pred.size(TensorDim.ENSEMBLE_DIM)] * pl_module.ens_comm_subgroup_size,
                    mgroup=pl_module.ens_comm_subgroup,
                )

            # Post-process for metrics (in physical space)
            post_processor = pl_module.model.post_processors[dataset_name]
            metrics_dict = pl_module.metrics[dataset_name]
            val_metric_ranges = pl_module.val_metric_ranges[dataset_name]

            # Gather the grid up front when any loss/metric does not support sharding, so non-sharding
            # metrics (e.g. spectral) get the full grid; gather commutes with post-processing below.
            pred, target, grid_shard_slice = pl_module._prepare_tensors_for_loss(
                pred,
                target,
                dataset_name=dataset_name,
                validation_mode=True,
            )

            for t in range(n_timesteps):
                # Slice single timestep: remove time dim
                pred_t = pred[:, t : t + 1, :, :, :]  # keep time dim for post-processor
                target_t = target[:, t : t + 1, :, :, :]

                pred_t_post = post_processor(pred_t, in_place=False)
                target_t_post = post_processor(target_t, in_place=False)

                for metric_name, metric in metrics_dict.items():
                    if not isinstance(metric, BaseLoss):
                        continue

                    for mkey, indices in val_metric_ranges.items():
                        step_name = f"val_{metric_name}_metric/{dataset_name}/{mkey}/t_{t + 1}"

                        metric_kwargs = {
                            "scaler_indices": (..., indices),
                            "without_scalers": [TensorDim.TIME],
                            "grid_shard_slice": grid_shard_slice,
                            "group": pl_module.model_comm_group,
                            "pred_layout": IndexSpace.MODEL_OUTPUT,
                            "target_layout": IndexSpace.DATA_FULL,
                        }
                        if getattr(metric, "needs_shard_layout_info", False):
                            # grid_shard_sizes must stay None once gathered, else an already-full tensor is re-sharded.
                            metric_kwargs.update(
                                grid_dim=pl_module.grid_dim,
                                grid_shard_sizes=(
                                    pl_module.grid_shard_sizes[dataset_name] if grid_shard_slice is not None else None
                                ),
                            )

                        value = metric(pred_t_post, target_t_post, **metric_kwargs)

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
