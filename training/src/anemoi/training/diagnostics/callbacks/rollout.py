# (C) Copyright 2024- Anemoi contributors.
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
        _ = checkpoint
        trainer.datamodule.rollout = pl_module.rollout
        trainer.datamodule.update_rollout()

    def on_train_epoch_end(self, trainer: pl.Trainer, *_) -> None:
        """
        Update the rollout values in the datamodule every training epoch.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer
        """
        trainer.datamodule.update_rollout()

    def on_train_batch_start(self, trainer: pl.Trainer, *_) -> None:
        """
        Update the rollout values in the datamodule every training batch.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer
        """
        trainer.datamodule.update_rollout()
