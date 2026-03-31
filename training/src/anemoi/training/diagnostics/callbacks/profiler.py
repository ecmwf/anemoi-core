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
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT

LOGGER = logging.getLogger(__name__)


class MemorySnapshotRecorder(Callback):
    """Record memory snapshot using torch.cuda._record_memory_history()."""

    def __init__(self, dirpath: str, steps: int, warmup: int = 0):
        """Initialise MemorySnapshotRecorder.

        Parameters
        ----------
        dirpath : str
            Directory to save the memory snapshot pickle file.
        steps : int
            Number of training steps to record after warmup.
        warmup : int, optional
            Number of warmup steps before recording starts, by default 0.
        """
        super().__init__()
        self.dirpath = Path(dirpath)
        self.warmup = warmup or 0
        self.num_steps = steps + self.warmup
        self.status = False

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        # Access batch_size from the dataloader directly
        train_dataloader = trainer.train_dataloader
        if isinstance(train_dataloader, list):
            train_dataloader = train_dataloader[0]
        batch_size = train_dataloader.batch_size
        assert self.num_steps % batch_size == 0, "Snapshot steps is not a multiple of batch size"
        assert self.warmup % batch_size == 0, "Snapshot warmup steps is not a multiple of batch size"

    @rank_zero_only
    def _start_snapshot_recording(self) -> None:
        LOGGER.info("Starting snapshot record_memory_history")
        torch.cuda.memory._record_memory_history()
        self.status = True

    @rank_zero_only
    def _save_snapshot(self) -> None:
        self.memory_snapshot_fname = self.dirpath / "memory_snapshot.pickle"
        try:
            LOGGER.info("Saving memory snapshot to %s", self.memory_snapshot_fname)
            torch.cuda.memory._dump_snapshot(f"{self.memory_snapshot_fname}")
        except BaseException:
            LOGGER.exception("Failed to capture memory snapshot")

    @rank_zero_only
    def stop_record_memory_history(self) -> None:
        LOGGER.info("Stopping snapshot record_memory_history")
        torch.cuda.memory._record_memory_history(enabled=None)

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, batch, batch_idx
        if trainer.global_step == self.warmup:
            self._start_snapshot_recording()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del batch, batch_idx, pl_module, outputs
        if trainer.global_step == self.num_steps:
            if self.status is True:
                self._save_snapshot()
                self.stop_record_memory_history()
            else:
                LOGGER.info("Snapshot recording was not started so no snapshot was saved")
