# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Dropout scheduler callback for curriculum-style training."""

import logging
import math
from typing import Any

import pytorch_lightning as pl

LOGGER = logging.getLogger(__name__)

SCHEDULES = ("cosine", "linear", "step")


class DropoutScheduler(pl.Callback):
    """Schedules the dropout probability of a spatial dropout preprocessor over training.

    Decays ``dropout_prob`` from ``start_prob`` to ``end_prob`` over ``total_steps``
    training steps using the specified schedule. This encourages the model to learn
    general representations early (high dropout) and refine with more data later
    (low dropout).

    The schedule is a pure function of ``trainer.global_step``, so the callback holds
    no checkpoint state and resuming picks the schedule up wherever the step count
    left off. Editing the schedule in the config and resuming therefore takes effect,
    rather than being overridden by a stale value restored from the checkpoint.

    ``RandomSpatialDropout`` is gated on ``self.training``, so the scheduled dropout
    applies to training only -- never to validation or inference.

    Parameters
    ----------
    processor_name : str
        Name of the preprocessor to target, e.g. "spatial_dropout" or "spatial_dropout2".
    start_prob : float
        Initial dropout probability at step 0.
    end_prob : float
        Final (floor) dropout probability after ``total_steps``.
    total_steps : int
        Number of training steps over which to decay.
    schedule : str
        Decay schedule: "cosine", "linear", or "step".
    step_milestones : list of float, optional
        For "step" schedule only: fractions of ``total_steps`` at which to reduce
        dropout, each strictly within (0, 1). E.g. [0.3, 0.6] drops at 30% and 60%
        of ``total_steps``, reaching ``end_prob`` at the last milestone.
    dataset_name : str
        Name of the dataset key (default: "data").
    """

    def __init__(
        self,
        processor_name: str = "spatial_dropout",
        start_prob: float = 0.7,
        end_prob: float = 0.2,
        total_steps: int = 50000,
        schedule: str = "cosine",
        step_milestones: list[float] | None = None,
        dataset_name: str = "data",
    ) -> None:
        super().__init__()

        for name, prob in (("start_prob", start_prob), ("end_prob", end_prob)):
            if not 0.0 <= prob <= 1.0:
                msg = f"{name} must be in [0, 1], got {prob}"
                raise ValueError(msg)
        if total_steps <= 0:
            msg = f"total_steps must be positive, got {total_steps}"
            raise ValueError(msg)
        if schedule not in SCHEDULES:
            msg = f"Unknown schedule {schedule!r}, expected one of {SCHEDULES}"
            raise ValueError(msg)

        # A milestone at 1.0 is unreachable: _compute_dropout returns end_prob once
        # global_step >= total_steps, so that stage would never be applied.
        milestones = sorted(step_milestones) if step_milestones is not None else [0.33, 0.66]
        if any(not 0.0 < milestone < 1.0 for milestone in milestones):
            msg = f"step_milestones must all be strictly within (0, 1), got {milestones}"
            raise ValueError(msg)

        self.processor_name = processor_name
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.total_steps = total_steps
        self.schedule = schedule
        self.step_milestones = milestones
        self.dataset_name = dataset_name

        self._processor = None

    def _compute_dropout(self, global_step: int) -> float:
        """Compute the dropout probability for the given global step.

        Parameters
        ----------
        global_step : int
            The current training step.

        Returns
        -------
        float
            The scheduled dropout probability.
        """
        if global_step >= self.total_steps:
            return self.end_prob

        progress = global_step / self.total_steps  # 0 -> 1
        span = self.start_prob - self.end_prob

        if self.schedule == "linear":
            return self.start_prob - span * progress

        if self.schedule == "cosine":
            # Cosine annealing: smooth decay
            return self.end_prob + span * 0.5 * (1 + math.cos(math.pi * progress))

        # Step schedule: piecewise constant, reaching end_prob at the last milestone
        n_stages = len(self.step_milestones)
        crossed = sum(progress >= milestone for milestone in self.step_milestones)
        return self.start_prob - span * crossed / n_stages

    def _resolve_processor(self, pl_module: pl.LightningModule) -> Any:
        """Look up the target preprocessor, raising if it is not usable.

        Parameters
        ----------
        pl_module : pl.LightningModule
            The training task holding the model.

        Returns
        -------
        Any
            The resolved preprocessor module.

        Raises
        ------
        ValueError
            If the dataset or processor is missing, or the processor has no
            variables to drop.
        """
        pre_processors = pl_module.model.pre_processors
        if self.dataset_name not in pre_processors:
            msg = (
                f"DropoutScheduler: dataset {self.dataset_name!r} not found in model pre_processors "
                f"(available: {list(pre_processors.keys())})"
            )
            raise ValueError(msg)

        processors = pre_processors[self.dataset_name].processors
        if self.processor_name not in processors:
            msg = (
                f"DropoutScheduler: processor {self.processor_name!r} not found in dataset "
                f"{self.dataset_name!r} (available: {list(processors.keys())})"
            )
            raise ValueError(msg)

        processor = processors[self.processor_name]

        # RandomSpatialDropout only registers dropout_indices when its configured
        # dropout_prob > 0, so a data config of 0.0 leaves nothing to drop and the
        # scheduler would run as a silent no-op.
        if len(getattr(processor, "dropout_indices", ())) == 0:
            msg = (
                f"DropoutScheduler: processor {self.processor_name!r} has no variables to drop. "
                f"Set a non-zero dropout_prob in the data config; the scheduler overwrites it from step 0."
            )
            raise ValueError(msg)

        return processor

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Resolve the target preprocessor before training begins.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer, not used.
        pl_module : pl.LightningModule
            The training task holding the model.
        stage : str
            The Lightning stage being set up.
        """
        del trainer
        if stage == "fit":
            self._processor = self._resolve_processor(pl_module)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log the schedule and apply the initial dropout probability.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer.
        pl_module : pl.LightningModule
            The training task holding the model.
        """
        if self._processor is None:
            self._processor = self._resolve_processor(pl_module)

        LOGGER.info(
            "DropoutScheduler: %s will decay from %.3f to %.3f over %d steps using the %s schedule.",
            self.processor_name,
            self.start_prob,
            self.end_prob,
            self.total_steps,
            self.schedule,
        )
        self._processor.dropout_prob = self._compute_dropout(trainer.global_step)

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update the dropout probability at the start of each training batch.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer.
        pl_module : pl.LightningModule
            The training task, used to log the current rate.
        batch : Any
            The current batch, not used.
        batch_idx : int
            Index of the current batch, not used.
        """
        del batch, batch_idx

        new_prob = self._compute_dropout(trainer.global_step)
        self._processor.dropout_prob = new_prob

        # Dropout is training-only, so this trace is the only evidence on the run
        # that the schedule is advancing.
        pl_module.log(
            f"dropout/{self.processor_name}",
            new_prob,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=pl_module.logger_enabled,
            sync_dist=False,
        )
