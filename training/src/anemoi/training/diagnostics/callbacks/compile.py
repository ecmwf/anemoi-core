# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Callbacks for persisting torch.compile artifacts."""

import logging

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from anemoi.training.utils.compile import load_compile_cache
from anemoi.training.utils.compile import save_compile_cache

LOGGER = logging.getLogger(__name__)


class CompileCache(Callback):
    """Save and load compile artifacts once.

    Artifacts are loaded at the start of the very first batch.
    Artifacts are saved once after the requested number of steps.
    """

    def __init__(self, compile_cache_file: str, save_after_steps: int = 1) -> None:
        super().__init__()
        self.compile_cache_file = compile_cache_file
        self.save_after_steps = save_after_steps
        self._saved = False
        self._loaded = False

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Load compile artifacts once at the start of the first training batch."""
        del pl_module, batch, batch_idx
        torch.compiler.cudagraph_mark_step_begin()

        if not self._loaded:
            LOGGER.info("Loading torch.compile cache from %s", self.compile_cache_file)
            load_compile_cache(self.compile_cache_file)
            self._loaded = True

        if self._saved or trainer.global_step < self.save_after_steps:
            return

        save_compile_cache(self.compile_cache_file)
        self._saved = True

    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Mark the start of a validation step for CUDA graph capture."""
        del trainer, pl_module, batch, dataloader_idx
        torch.compiler.cudagraph_mark_step_begin()

        if batch_idx == self.save_after_steps:
            LOGGER.info("Saving torch.compile cache to %s", self.compile_cache_file)
            save_compile_cache(self.compile_cache_file)
