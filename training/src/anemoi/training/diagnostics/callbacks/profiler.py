# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# ruff: noqa: ANN001


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

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dirpath = Path(self.config.hardware.paths.profiler)

        self.warmup = self.config.diagnostics.benchmark_profiler.snapshot.warmup
        if not self.warmup:
            self.warmup = 0
        self.num_steps = (
            self.config.diagnostics.benchmark_profiler.snapshot.steps + self.warmup
        )  # be consistent with profiler scheduler
        self.status = False

        assert (
            self.num_steps % self.config.dataloader.batch_size.training == 0
        ), "Snapshot steps is not a multiple of batch size"
        assert (
            self.warmup % self.config.dataloader.batch_size.training == 0
        ), "Snapshot Warmup steps is not a multiple of batch size"

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
