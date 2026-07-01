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
        self._last_increased_epoch: int = -1

    def should_increase(self, current_epoch: int) -> bool:
        """Check if rollout should be increased at the end of the current epoch."""
        return (
            self.epoch_increment > 0
            and current_epoch % self.epoch_increment == 0
            and self.step < self.maximum
            and current_epoch != self._last_increased_epoch
        )

    def increase(self, current_epoch: int) -> None:
        """Increase the rollout window by one step."""
        if self.step < self.maximum:
            self.step += 1
            self._last_increased_epoch = current_epoch
            LOGGER.info("Rollout window length has been increased to %d.", self.step)

    def state_dict(self) -> dict:
        """Return serialisable state."""
        return {"step": self.step, "last_increased_epoch": self._last_increased_epoch}

    def load_state_dict(self, state: dict) -> None:
        """Restore state from a dict produced by :meth:`state_dict`."""
        self.step = state["step"]
        self._last_increased_epoch = state["last_increased_epoch"]


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
        **kwargs,
    ) -> None:
        self.timestep = frequency_to_timedelta(timestep)
        self.num_input_steps = multistep_input
        self.num_output_steps = multistep_output
        self.rollout = RolloutConfig(**(rollout or {}))
        self.validation_rollout = validation_rollout

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

    def steps(self, mode: str = "training") -> tuple[dict[str, int], ...]:
        """Return the current steps configuration based on the rollout step."""
        max_rollout = self.rollout.step
        if mode == "validation" and self.validation_rollout is not None:
            max_rollout = self.validation_rollout
        return tuple({"rollout_step": i} for i in range(max_rollout))

    def get_metric_name(self, rollout_step: int = 0, **_kwargs) -> str:
        """Get the metric name for the current step."""
        return f"_rstep{rollout_step}"

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
            rollout_step = self.rollout.step
        elif mode == "validation":
            rollout_step = self.rollout.step if self.validation_rollout is None else self.validation_rollout
        else:
            LOGGER.debug(
                "Unknown mode '%s' for %s.get_offsets(); using offsets for the longest configured rollout.",
                mode,
                self.__class__.__name__,
            )
            validation_rollout = self.rollout.maximum if self.validation_rollout is None else self.validation_rollout
            rollout_step = max(self.rollout.maximum, validation_rollout)

        return self._compute_rollout_offsets(rollout_step)

    def get_output_offsets(
        self,
        rollout_step: int = 0,
        **_kwargs,
    ) -> list[datetime.timedelta]:
        """Return output offsets shifted by ``rollout_step``."""
        shift = self._step_shift * rollout_step
        return sorted(o + shift for o in self._output_offsets)

    def _requested_output_relative_times(
        self,
        dataset_name: str,
        rollout_step: int = 0,
        step: int | None = None,
    ) -> list[int]:
        rollout_step = rollout_step if step is None else step
        requested = self.dataset_target_relative_times_by_dataset.get(dataset_name)
        if requested is not None:
            if len(requested) == 0:
                return []
            rollout_window = max(self.rollout.maximum, self.validation_rollout)
            if len(requested) % rollout_window != 0:
                msg = (
                    f"Dataset '{dataset_name}' target indices {requested} do not divide evenly across "
                    f"rollout window {rollout_window}."
                )
                raise ValueError(msg)

            per_step_output = len(requested) // rollout_window
            start = rollout_step * per_step_output
            stop = start + per_step_output
            return requested[start:stop]

        return super()._requested_output_relative_times(dataset_name, rollout_step=rollout_step)

    def _prepare_next_input_steps(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        batch: torch.Tensor,
        dataset_name: str | None,
        rollout_step: int,
        data_indices: IndexCollection,
    ) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor, int | None]], bool]:
        use_dataset_time_maps = len(self.dataset_time_maps) > 0
        if use_dataset_time_maps:
            if dataset_name is None:
                msg = "`dataset_name` is required for mixed-frequency input updates."
                raise ValueError(msg)

            requested_output_relative_times = self._requested_output_relative_times(
                dataset_name,
                rollout_step=rollout_step,
            )
            step_specs = []
            for relative_time in [
                int(relative_time + (rollout_step + 1) * self.num_output_timesteps)
                for relative_time in self._requested_input_relative_times(dataset_name)
            ]:
                batch_position = self._sample_batch_position(dataset_name=dataset_name, relative_time=relative_time)
                if not 0 <= batch_position < batch.shape[1]:
                    msg = (
                        f"Mixed-frequency input update for dataset '{dataset_name}' "
                        f"resolved relative time {relative_time} "
                        f"to batch position {batch_position}, but batch only has {batch.shape[1]} time steps."
                    )
                    raise ValueError(msg)
                x_step = batch[:, batch_position, ..., data_indices.data.input.full].clone()

                if x_step.shape[1] == 1 and y_pred.shape[2] != 1:
                    x_step = x_step.expand(-1, y_pred.shape[2], -1, -1).clone()

                pred_position = None
                if int(relative_time) in requested_output_relative_times:
                    pred_position = requested_output_relative_times.index(int(relative_time))
                step_specs.append((batch_position, x_step, pred_position))

            return x, step_specs, True

        keep_steps = min(self.num_input_steps, self.num_output_steps)
        x = x.roll(-keep_steps, dims=1)

        # Compute batch indices for the output offsets of this rollout step
        output_batch_indices = self.get_batch_output_indices(rollout_step=rollout_step)
        step_specs = [(output_batch_indices[-(i + 1)], x[:, -(i + 1)], -(i + 1)) for i in range(keep_steps)]
        return x, step_specs, False

    def _advance_dataset_input(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        batch: torch.Tensor,
        dataset_name: str | None = None,
        rollout_step: int = 0,
        data_indices: IndexCollection | None = None,
        output_mask: object | None = None,
        grid_shard_slice: slice | None = None,
    ) -> torch.Tensor:
        """Advance a single dataset's input state for the next rollout step.

        Supports model outputs shaped like ``(B, T, E, G, V)``.
        """
        x, step_specs, use_dataset_time_maps = self._prepare_next_input_steps(
            x,
            y_pred,
            batch,
            dataset_name,
            rollout_step,
            data_indices,
        )
        next_steps = []
        for i, (batch_position, x_step, pred_position) in enumerate(step_specs):
            # Get prognostic variables
            if pred_position is not None:
                x_step[..., data_indices.model.input.prognostic] = y_pred[
                    :,
                    pred_position,
                    ...,
                    data_indices.model.output.prognostic,
                ]

            true_state = batch[:, batch_position]
            if true_state.shape[1] == 1 and x_step.shape[1] != 1:
                true_state = true_state.expand(-1, x_step.shape[1], -1, -1)

            if output_mask is not None:
                x_step = output_mask.rollout_boundary(
                    x_step,
                    true_state,
                    data_indices,
                    grid_shard_slice=grid_shard_slice,
                )

            # get new "constants" needed for time-varying fields
            forcing = batch[:, batch_position, ..., data_indices.data.input.forcing]
            if forcing.shape[1] == 1 and x_step.shape[1] != 1:
                forcing = forcing.expand(-1, x_step.shape[1], -1, -1)
            x_step[..., data_indices.model.input.forcing] = forcing

            if use_dataset_time_maps:
                next_steps.append(x_step)
            else:
                x[:, -(i + 1)] = x_step

        if use_dataset_time_maps:
            return torch.stack(next_steps, dim=1)
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
                dataset_name=dataset_name,
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
            on_step=False,
            on_epoch=True,
            logger=logger_enabled,
            rank_zero_only=True,
            sync_dist=False,
        )

    def training_runtime_state_dict(self) -> dict:
        """Return training runtime state to be persisted in the training checkpoint.

        Captures the current rollout curriculum step so that job resume
        continues the schedule from where it left off rather than restarting
        from ``rollout.start``.
        """
        return {"rollout": self.rollout.state_dict()}

    def load_training_runtime_state_dict(self, state: dict) -> None:
        """Restore training runtime state from a training checkpoint."""
        if "rollout" in state:
            self.rollout.load_state_dict(state["rollout"])

    def on_train_epoch_end(self, current_epoch: int) -> None:
        if self.rollout.should_increase(current_epoch):
            self.rollout.increase(current_epoch)

    def _get_timestep_for_metadata(self) -> str:
        """Get the timestep string for metadata."""
        return frequency_to_string(self.timestep)
