# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Callback scheduling the weight of one CombinedLoss component over training steps."""

import logging
import math
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

LOGGER = logging.getLogger(__name__)

SCHEDULES = ("linear", "cosine")


class LossWeightScheduler(Callback):
    """Ramp the weight of one ``CombinedLoss`` component between two values.

    Useful to phase in an observation-operator term (e.g. ``RefractivityOperatorLoss``)
    only after the learning-rate warm-up and the first epochs have shaped the state, so
    its large early residuals cannot destabilise the optimiser. Before ``start_step`` the
    weight is ``start_weight``; it then follows ``schedule`` over ``ramp_steps`` steps to
    ``end_weight`` and stays there. The schedule is a pure function of
    ``trainer.global_step``, so it is consistent across restarts.

    Parameters
    ----------
    loss_index : int
        Position of the component in ``CombinedLoss.losses`` (the order of ``losses`` in
        the training-loss config).
    start_weight, end_weight : float
        Weights before the ramp and after it.
    start_step : int
        Global step at which the ramp begins.
    ramp_steps : int
        Length of the ramp in steps.
    schedule : {"linear", "cosine"}
        Ramp shape.
    dataset_name : str
        Dataset whose training loss is scheduled.
    log_every_n_steps : int
        Logging frequency of the current weight (``loss_weight/<dataset>/<index>``).
    """

    def __init__(
        self,
        loss_index: int,
        start_weight: float = 0.0,
        end_weight: float = 1.0,
        start_step: int = 0,
        ramp_steps: int = 1,
        schedule: str = "linear",
        dataset_name: str = "data",
        log_every_n_steps: int = 100,
    ) -> None:
        super().__init__()
        if loss_index < 0:
            msg = f"loss_index must be non-negative, got {loss_index}"
            raise ValueError(msg)
        if start_step < 0 or ramp_steps <= 0:
            msg = f"start_step must be >= 0 and ramp_steps > 0, got {start_step}, {ramp_steps}"
            raise ValueError(msg)
        if schedule not in SCHEDULES:
            msg = f"Unknown schedule {schedule!r}, expected one of {SCHEDULES}"
            raise ValueError(msg)
        self.loss_index = loss_index
        self.start_weight = float(start_weight)
        self.end_weight = float(end_weight)
        self.start_step = int(start_step)
        self.ramp_steps = int(ramp_steps)
        self.schedule = schedule
        self.dataset_name = dataset_name
        self.log_every_n_steps = max(1, int(log_every_n_steps))

    def weight_at(self, global_step: int) -> float:
        """Scheduled weight at a global step."""
        if global_step <= self.start_step:
            return self.start_weight
        progress = min(1.0, (global_step - self.start_step) / self.ramp_steps)
        if self.schedule == "cosine":
            progress = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.start_weight + (self.end_weight - self.start_weight) * progress

    def _combined_loss(self, pl_module: pl.LightningModule) -> Any:
        losses = getattr(pl_module, "loss", None)
        loss = losses[self.dataset_name] if losses is not None and self.dataset_name in losses else None
        if loss is None or not hasattr(loss, "loss_weights"):
            msg = (
                f"LossWeightScheduler: training loss for dataset {self.dataset_name!r} is not a CombinedLoss "
                "(no loss_weights attribute)."
            )
            raise TypeError(msg)
        if self.loss_index >= len(loss.loss_weights):
            msg = (
                f"LossWeightScheduler: loss_index {self.loss_index} out of range "
                f"for {len(loss.loss_weights)} components"
            )
            raise IndexError(msg)
        return loss

    def apply(self, pl_module: pl.LightningModule, global_step: int) -> float:
        """Set the scheduled weight on the combined loss and return it."""
        loss = self._combined_loss(pl_module)
        weight = self.weight_at(global_step)
        weights = list(loss.loss_weights)
        weights[self.loss_index] = weight
        loss.loss_weights = tuple(weights)
        return weight

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        weight = self.apply(pl_module, trainer.global_step)
        LOGGER.info(
            "LossWeightScheduler: component %d of dataset %r starts at %.4g (ramp %d -> %d steps to %.4g, %s)",
            self.loss_index,
            self.dataset_name,
            weight,
            self.start_step,
            self.start_step + self.ramp_steps,
            self.end_weight,
            self.schedule,
        )

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        weight = self.apply(pl_module, trainer.global_step)
        if trainer.global_step % self.log_every_n_steps == 0:
            pl_module.log(
                f"loss_weight/{self.dataset_name}/{self.loss_index}",
                weight,
                on_step=True,
                on_epoch=False,
                logger=getattr(pl_module, "logger_enabled", True),
                rank_zero_only=True,
            )
