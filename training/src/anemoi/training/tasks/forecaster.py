# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
import random
from collections.abc import Callable

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.diagnostics.callbacks.plot_adapter import ForecasterPlotAdapter
from anemoi.training.tasks.base import BaseTask
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

LOGGER = logging.getLogger(__name__)


class RolloutConfig:
    """Rollout configuration for autoregressive training."""

    def __init__(self, start: int = 1, epoch_increment: int = 0, maximum: int = 1) -> None:
        """Initialize rollout configuration."""
        self.start = start
        self.epoch_increment = epoch_increment
        self.maximum = maximum
        self.step = self.start

    def should_increase(self, current_epoch: int) -> bool:
        """Check if rollout should be increased at the end of the current epoch."""
        return self.epoch_increment > 0 and current_epoch % self.epoch_increment == 0 and self.step < self.maximum

    def increase(self) -> None:
        """Increase the rollout window by one step."""
        if self.step < self.maximum:
            self.step += 1
            LOGGER.info("Rollout window length has been increased to %d.", self.step)


class Forecaster(BaseTask):
    """Forecasting task implementation.

    Builds input and output offsets from ``multistep_input``,
    ``multistep_output`` and a ``timestep`` string (e.g. ``"6H"``).

    For rollout training, training offsets extend up to the current
    ``rollout.step`` so the dataloader only loads the required time
    steps. ``rollout.step`` grows via ``on_train_epoch_end``.
    """

    name: str = "forecaster"

    def __init__(
        self,
        multistep_input: int,
        multistep_output: int,
        timestep: str,
        rollout: dict | None = None,
        validation_rollout: int | None = None,
        rollout_loss_weights: list[float] | None = None,
        log_rollout_step_losses: bool = False,
        pushforward: dict | None = None,
        **kwargs,
    ) -> None:

        self.timestep = frequency_to_timedelta(timestep)
        self.num_input_steps = multistep_input
        self.num_output_steps = multistep_output
        self.rollout = RolloutConfig(**(rollout or {}))
        self.validation_rollout = validation_rollout
        self.rollout_loss_weights = None if rollout_loss_weights is None else [float(w) for w in rollout_loss_weights]
        self.log_rollout_step_losses = log_rollout_step_losses
        self.pushforward = pushforward
        if self.pushforward is not None:
            LOGGER.info("Pushforward rollouts (training only): %s", self.pushforward)
        if self.rollout_loss_weights is not None:
            assert all(w > 0 for w in self.rollout_loss_weights), "rollout_loss_weights must be positive"
            assert len(self.rollout_loss_weights) >= self.max_training_rollout, (
                f"rollout_loss_weights has {len(self.rollout_loss_weights)} entries "
                f"but the longest training rollout is {self.max_training_rollout}"
            )
            LOGGER.info("Rollout loss weights (training only, per step): %s", self.rollout_loss_weights)

        if len(kwargs) > 0:
            LOGGER.warning(
                "The following extra parameters were provided to %s but will be ignored: %s",
                self.__class__.__name__,
                kwargs,
            )

        # Input: e.g. multistep_input=2, timestep=6H     ->  [-6H, 0H]
        input_offsets = [-1 * i * self.timestep for i in range(multistep_input)]
        # Outputs: e.g. multistep_output=1, timestep=6H  -> [[6H], [12H], [18H], ...] up to rollout.maximum
        output_offsets = [(i + 1) * self.timestep for i in range(multistep_output)]
        super().__init__(input_offsets=input_offsets, output_offsets=output_offsets)
        self._plot_adapter = ForecasterPlotAdapter(self)

    @property
    def max_training_rollout(self) -> int:
        """Longest training rollout: the largest pushforward length, else ``rollout.maximum``."""
        return max(self.pushforward["lengths"]) if self.pushforward is not None else self.rollout.maximum

    def steps(self, mode: str = "training", seed: int | None = None) -> tuple[dict[str, int], ...]:
        """Return the current steps configuration based on the rollout step.

        With ``pushforward``, each training rollout has a length drawn from ``pushforward.lengths``
        with ``seed`` (the global step, so that all ranks draw the same length). Without a seed the
        longest length is returned.
        """
        max_rollout = self.rollout.step
        if mode == "validation" and self.validation_rollout is not None:
            max_rollout = self.validation_rollout
        if mode == "training" and self.pushforward is not None:
            lengths = self.pushforward["lengths"]
            # a seeded, reproducible draw (not cryptographic)
            max_rollout = max(lengths) if seed is None else random.Random(seed).choice(lengths)  # noqa: S311
        return tuple({"rollout_step": i} for i in range(max_rollout))

    def get_metric_name(self, rollout_step: int = 0, **_kwargs) -> str:
        """Get the metric name for the current step."""
        return f"_rstep{rollout_step}"

    def step_loss_weights(self, task_steps: tuple[dict, ...], mode: str = "training") -> list[float]:
        """Per-step loss weights from ``rollout_loss_weights`` and ``pushforward``, in training only.

        With ``pushforward``, only the first ``anchor_steps`` and the last ``grad_steps`` steps get a
        loss; the steps in between get weight 0 and run without gradients. Weights are renormalised
        to sum to the number of steps, so they set relative contributions without changing the
        overall loss scale. Validation always uses the plain average so validation losses stay
        comparable between runs.
        """
        if mode != "training" or (self.rollout_loss_weights is None and self.pushforward is None):
            return [1.0] * len(task_steps)
        n_steps = len(task_steps)
        weights = [
            1.0 if self.rollout_loss_weights is None else self.rollout_loss_weights[kwargs["rollout_step"]]
            for kwargs in task_steps
        ]
        if self.pushforward is not None:
            first, last = self.pushforward.get("anchor_steps", 0), self.pushforward.get("grad_steps", 1)
            weights = [w if (i < first or i >= n_steps - last) else 0.0 for i, w in enumerate(weights)]
        scale = n_steps / sum(weights)
        return [w * scale for w in weights]

    @property
    def _step_shift(self) -> datetime.timedelta:
        """Time shift between consecutive rollout steps."""
        return self.timestep * self.num_output_steps

    def _compute_rollout_offsets(self, rollout_step: int) -> list[datetime.timedelta]:
        """Compute the full list of offsets needed for the current rollout configuration."""
        all_offsets = set(self._input_offsets)
        for step in range(rollout_step):
            shift = self._step_shift * step
            for o in self._output_offsets:
                all_offsets.add(o + shift)
        return sorted(all_offsets)

    def get_offsets(self, mode: str | None = None) -> list[datetime.timedelta]:
        if mode == "training":
            rollout_step = self.rollout.step if self.pushforward is None else self.max_training_rollout
        elif mode == "validation":
            rollout_step = self.rollout.step if self.validation_rollout is None else self.validation_rollout
        else:
            LOGGER.debug(
                "Unknown mode '%s' for %s.get_offsets(); using offsets for the longest configured rollout.",
                mode,
                self.__class__.__name__,
            )
            validation_rollout = self.rollout.maximum if self.validation_rollout is None else self.validation_rollout
            rollout_step = max(self.max_training_rollout, validation_rollout)

        return self._compute_rollout_offsets(rollout_step)

    def get_output_offsets(
        self,
        rollout_step: int = 0,
        **_kwargs,
    ) -> list[datetime.timedelta]:
        """Return output offsets shifted by ``rollout_step``."""
        shift = self._step_shift * rollout_step
        return sorted(o + shift for o in self._output_offsets)

    def _advance_dataset_input(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        batch: torch.Tensor,
        rollout_step: int = 0,
        data_indices: IndexCollection | None = None,
        output_mask: object | None = None,
        grid_shard_slice: slice | None = None,
    ) -> torch.Tensor:
        """Advance a single dataset's input state for the next rollout step.

        Supports model outputs shaped like ``(B, T, E, G, V)``.
        """
        keep_steps = min(self.num_input_steps, self.num_output_steps)

        x = x.roll(-keep_steps, dims=1)

        # Compute batch indices for the output offsets of this rollout step
        output_batch_indices = self.get_batch_output_indices(rollout_step=rollout_step)

        for i in range(keep_steps):
            # Get prognostic variables
            x[:, -(i + 1), ..., data_indices.model.input.prognostic] = y_pred[
                :,
                -(i + 1),
                ...,
                data_indices.model.output.prognostic,
            ]

            batch_time_index = output_batch_indices[-(i + 1)]
            true_state = batch[:, batch_time_index]

            if output_mask is not None and true_state.shape[1] == 1 and x[:, -(i + 1)].shape[1] != 1:
                true_state = true_state.expand(-1, x[:, -(i + 1)].shape[1], -1, -1)

            x[:, -(i + 1)] = output_mask.rollout_boundary(
                x[:, -(i + 1)],
                true_state,
                data_indices,
                grid_shard_slice=grid_shard_slice,
            )

            # get new "constants" needed for time-varying fields
            x[:, -(i + 1), ..., data_indices.model.input.forcing] = batch[
                :,
                batch_time_index,
                ...,
                data_indices.data.input.forcing,
            ]
        return x

    def advance_input(
        self,
        x: dict[str, torch.Tensor],
        y_pred: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        rollout_step: int = 0,
        data_indices: dict[str, IndexCollection] | None = None,
        output_mask: dict[str, object] | None = None,
        grid_shard_slice: dict[str, slice | None] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Advance the input state for the next rollout step."""
        for dataset_name in x:
            x[dataset_name] = self._advance_dataset_input(
                x[dataset_name],
                y_pred[dataset_name],
                batch[dataset_name],
                rollout_step=rollout_step,
                data_indices=data_indices[dataset_name],
                output_mask=None if output_mask is None else output_mask[dataset_name],
                grid_shard_slice=None if grid_shard_slice is None else grid_shard_slice[dataset_name],
            )
        return x

    def log_extra(self, logger: Callable, logger_enabled: bool) -> None:
        """Log any task-specific information."""
        logger(
            "rollout",
            float(self.rollout.step),
            on_step=True,
            logger=logger_enabled,
            rank_zero_only=True,
            sync_dist=False,
        )

    def on_train_epoch_end(self, current_epoch: int) -> None:
        if self.rollout.should_increase(current_epoch):
            self.rollout.increase()

    def _get_timestep_for_metadata(self) -> str:
        """Get the timestep string for metadata."""
        return frequency_to_string(self.timestep)
