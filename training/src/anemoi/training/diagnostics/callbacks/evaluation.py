# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from contextlib import nullcontext

import pytorch_lightning as pl
import torch
from omegaconf import ListConfig
from pytorch_lightning.callbacks import Callback

LOGGER = logging.getLogger(__name__)


class RolloutEval(Callback):
    """Evaluates the model performance over a (longer) rollout window.

    Health warning: this callback runs only every ``every_n_batches`` validation batches,
    so metrics are a sampled view of validation dates. Metrics are logged with
    distributed synchronization.
    """

    def __init__(self, rollout: list[int | None] | ListConfig, every_n_batches: int) -> None:
        """Initialize RolloutEval callback.

        Parameters
        ----------
        rollout : list[int | None] | ListConfig
            Rollout lengths for evaluation. ``[None]`` follows the task validation rollout.
        every_n_batches : int
            Frequency of rollout evaluation, runs every `n` validation batches

        """
        super().__init__()

        assert isinstance(rollout, list | ListConfig), f"rollout must be a list of ints or None, got {type(rollout)}"
        rollout_values = list(rollout)

        LOGGER.debug(
            "Setting up RolloutEval callback with rollout = %s, every_n_batches = %d ...",
            rollout_values,
            every_n_batches,
        )
        self.rollout = rollout_values
        self.follow_task_validation_rollout = False
        if rollout_values == [None]:
            self.follow_task_validation_rollout = True
            self.max_rollout = None
        else:
            self.max_rollout = max(rollout_values)
        self.every_n_batches = every_n_batches

    def _eval(
        self,
        pl_module: pl.LightningModule,
        batch: dict[str, torch.Tensor],
    ) -> None:
        batch_tensor = batch
        if isinstance(batch, dict):
            batch_tensor = next(iter(batch.values()))

        if self.follow_task_validation_rollout:
            self.max_rollout = len(tuple(pl_module.task.steps("validation")))

        assert batch_tensor.shape[1] >= self.max_rollout * pl_module.n_step_output + pl_module.n_step_input, (
            "Batch length not sufficient for requested validation rollout length! "
            f"Set `task.validation_rollout` to at least {self.max_rollout}"
        )

        # NOTE: The configured rollout must be lower than or equal to `task.validation_rollout`,
        # because `_step(..., validation_mode=True)` uses the task setting to determine step count.
        with torch.no_grad():
            step_output = pl_module._step(batch, validation_mode=True)
            self._log(pl_module, step_output.loss, step_output.metrics, batch_tensor.shape[0])

    def _log(self, pl_module: pl.LightningModule, loss: torch.Tensor, metrics: dict, bs: int) -> None:

        loss_scales = loss
        loss = loss_scales.sum()
        loss_name = getattr(pl_module.loss, "name", pl_module.loss.__class__.__name__.lower())
        pl_module.log(
            f"val_r{self.max_rollout}_{loss_name}",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=False,
            logger=pl_module.logger_enabled,
            batch_size=bs,
            sync_dist=True,
        )
        if loss_scales.numel() > 1:
            for scale in range(loss_scales.numel()):
                pl_module.log(
                    f"val_r{self.max_rollout}_{loss_name}_{scale}",
                    loss_scales[scale],
                    on_epoch=True,
                    on_step=True,
                    prog_bar=False,
                    logger=pl_module.logger_enabled,
                    batch_size=bs,
                    sync_dist=True,
                )

        for mname, mvalue in metrics.items():
            for scale in range(mvalue.numel()):

                log_val = mvalue[scale] if mvalue.numel() > 1 else mvalue

                pl_module.log(
                    f"val_r{self.max_rollout}_" + mname + "_scale_" + str(scale),
                    log_val,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                    logger=pl_module.logger_enabled,
                    batch_size=bs,
                    sync_dist=True,
                )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        del outputs  # outputs are not used
        if batch_idx % self.every_n_batches == 0:
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
            # 'torch.compile.set_stance' tells the compiler to try use compiled code if it exists
            # but fall back to eager if it doesn't.
            # This is used because the evaluationRollout callback seems to introduce many different input shapes
            # These all force recompilation which slows down evaluation and eventually leads to a an error
            # once the config.model.recompile_limit is reached.
            with context and torch.compiler.set_stance("eager_on_recompile"):
                self._eval(pl_module, batch)
