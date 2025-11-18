# (C) Copyright 2024 Anemoi contributors.
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
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback

LOGGER = logging.getLogger(__name__)


class RolloutEval(Callback):
    """Evaluates the model performance over a (longer) rollout window."""

    def __init__(self, config: OmegaConf, rollout: int, every_n_batches: int) -> None:
        """Initialize RolloutEval callback.

        Parameters
        ----------
        config : dict
            Dictionary with configuration settings
        rollout : int
            Rollout length for evaluation
        every_n_batches : int
            Frequency of rollout evaluation, runs every `n` validation batches

        """
        super().__init__()
        self.config = config

        LOGGER.debug(
            "Setting up RolloutEval callback with rollout = %d, every_n_batches = %d ...",
            rollout,
            every_n_batches,
        )
        self.rollout = rollout
        self.every_n_batches = every_n_batches

    def _eval(
        self,
        pl_module: pl.LightningModule,
        batch: torch.Tensor,
    ) -> None:
        loss = torch.zeros(1, dtype=batch.dtype, device=pl_module.device, requires_grad=False)
        metrics = {}

        assert batch.shape[1] >= self.rollout + pl_module.multi_step, (
            "Batch length not sufficient for requested validation rollout length! "
            f"Set `dataloader.validation_rollout` to at least {max(self.rollout)}"
        )

        with torch.no_grad():
            for loss_next, metrics_next, _ in pl_module.rollout_step(
                batch,
                rollout=self.rollout,
                validation_mode=True,
            ):
                loss += loss_next
                metrics.update(metrics_next)

            # scale loss
            loss *= 1.0 / self.rollout
            self._log(pl_module, loss, metrics, batch.shape[0])

    def _log(self, pl_module: pl.LightningModule, loss: torch.Tensor, metrics: dict, bs: int) -> None:
        pl_module.log(
            f"val_r{self.rollout}_{getattr(pl_module.loss, 'name', pl_module.loss.__class__.__name__.lower())}",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=False,
            logger=pl_module.logger_enabled,
            batch_size=bs,
            sync_dist=False,
            rank_zero_only=True,
        )
        for mname, mvalue in metrics.items():
            pl_module.log(
                f"val_r{self.rollout}_" + mname,
                mvalue,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=pl_module.logger_enabled,
                batch_size=bs,
                sync_dist=False,
                rank_zero_only=True,
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
            batch = pl_module.allgather_batch(batch)

            precision_mapping = {
                "16-mixed": torch.float16,
                "bf16-mixed": torch.bfloat16,
            }
            prec = trainer.precision
            dtype = precision_mapping.get(prec)
            context = torch.autocast(device_type=batch.device.type, dtype=dtype) if dtype is not None else nullcontext()

            with context:
                self._eval(pl_module, batch)


class RolloutEvalEns(RolloutEval):
    """Evaluates the model performance over a (longer) rollout window."""

    def _eval(self, pl_module: pl.LightningModule, batch: torch.Tensor) -> None:
        """Rolls out the model and calculates the validation metrics.

        Parameters
        ----------
        pl_module : pl.LightningModule
            Lightning module object
        batch: torch.Tensor
            Batch tensor (bs, input_steps + forecast_steps, latlon, nvar)
        """
        loss = torch.zeros(1, dtype=batch.dtype, device=pl_module.device, requires_grad=False)
        assert (
            batch.shape[1] >= self.rollout + pl_module.multi_step
        ), f"Batch length ({batch.shape[1]}) not sufficient for requested rollout length!"

        metrics = {}
        with torch.no_grad():
            for loss_next, metrics_next, _, _ in pl_module.rollout_step(
                batch=batch,
                rollout=self.rollout,
                validation_mode=True,
            ):
                loss += loss_next
                metrics.update(metrics_next)

            # scale loss
            loss *= 1.0 / self.rollout
            self._log(pl_module, loss, metrics, batch.shape[0])

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        del outputs  # outputs are not used
        if batch_idx % self.every_n_batches == 0 and pl_module.ens_comm_group_rank == 0:
            precision_mapping = {
                "16-mixed": torch.float16,
                "bf16-mixed": torch.bfloat16,
            }
            prec = trainer.precision
            dtype = precision_mapping.get(prec)
            context = torch.autocast(device_type=batch.device.type, dtype=dtype) if dtype is not None else nullcontext()

            with context:
                self._eval(pl_module, batch)
