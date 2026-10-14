# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Per-rank host timestamps of every training step, for finding stragglers in model-parallel runs."""

import gc
import os
import sys
import time
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class StepTimingLogger(Callback):
    """Print one line per training step on every rank.

    The line holds wall-clock timestamps (``time.time()``) taken on the host when the batch is ready
    (``on_train_batch_start``), before the backward pass and at the end of the step. No device
    synchronisation is added. Comparing the timestamps of the same step across ranks shows which
    ranks arrive late, and whether that is spent waiting for data or in the step itself.
    """

    def __init__(self) -> None:
        super().__init__()
        self._t_start = 0.0
        self._t_backward = 0.0
        self._t_prev_end = None
        self._t_fwd = float("nan")
        self._hooked = False
        # Python garbage-collector time per step (all generations, and gen-2 = full collections)
        self._gc_t0 = 0.0
        self._gc_ms = 0.0
        self._gc2_ms = 0.0
        gc.callbacks.append(self._gc_event)

    def _gc_event(self, phase: str, info: dict) -> None:
        if phase == "start":
            self._gc_t0 = time.perf_counter()
            return
        dt = (time.perf_counter() - self._gc_t0) * 1e3
        self._gc_ms += dt
        if info.get("generation") == 2:
            self._gc2_ms += dt

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del trainer, pl_module
        # ANEMOI_GC_MODE=freeze: collect once, then move every surviving object into the permanent generation so
        # later full collections only scan objects created during training
        if os.environ.get("ANEMOI_GC_MODE", "") == "freeze":
            gc.collect()
            gc.freeze()
            sys.stdout.write(f"[steptime] gc frozen: {gc.get_freeze_count()} objects\n")

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int) -> None:
        del batch, batch_idx
        self._t_start = time.time()
        if not self._hooked:
            # host time when the model forward returns, i.e. before the loss and its reductions/logging syncs
            pl_module.model.register_forward_hook(lambda *_: setattr(self, "_t_fwd", time.time()))
            self._hooked = True

    def on_before_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule, loss: Any) -> None:
        del trainer, pl_module, loss
        self._t_backward = time.time()

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        del pl_module, outputs, batch
        t_end = time.time()
        data_wait = self._t_start - self._t_prev_end if self._t_prev_end is not None else float("nan")
        sys.stdout.write(
            f"[steptime] rank={trainer.global_rank} step={batch_idx} start={self._t_start:.6f} "
            f"backward={self._t_backward:.6f} end={t_end:.6f} data_wait={data_wait:.6f} fwd_end={self._t_fwd:.6f} "
            f"gc_ms={self._gc_ms:.1f} gc2_ms={self._gc2_ms:.1f}\n"
        )
        sys.stdout.flush()
        self._t_prev_end = t_end
        self._gc_ms = self._gc2_ms = 0.0
