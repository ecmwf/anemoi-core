# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging

import pytorch_lightning as pl

LOGGER = logging.getLogger(__name__)


class UpdateRollout(pl.callbacks.Callback):
    """Update Rollout values in datamodule."""

    def __init__(self) -> None:
        super().__init__()

    def _update_rollout(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        epoch: int | None = None,
        step: int | None = None,
    ) -> None:
        rollsched = pl_module.rollout
        with rollsched.at(epoch=epoch, step=step):
            rollout = rollsched.current_maximum

        LOGGER.debug("Propagating rollout value %s to datamodule", rollout)
        trainer.datamodule.update_rollout(rollout=rollout)

    def on_load_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict) -> None:
        """
        Update the rollout values in the datamodule when loading a checkpoint.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer
        pl_module : pl.LightningModule
            Model
        checkpoint : dict
            Checkpoint dictionary
        """
        self._update_rollout(trainer, pl_module, epoch=checkpoint["epoch"], step=checkpoint["global_step"])

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *_) -> None:
        """
        Update the rollout values in the datamodule every validation epoch.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer
        pl_module : pl.LightningModule
            Model
        """
        if trainer.sanity_checking:
            return

        # Offset of 1 needed as the epoch counter does not increment
        # until after the epoch ends.
        self._update_rollout(trainer, pl_module, epoch=trainer.current_epoch + 1)
