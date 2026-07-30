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
        input_offsets: list[str] | None = None,
        output_offsets: list[str] | None = None,
        rollout_shift: str = "0H",
        consistency_check: bool = True,
        multistep_input: int | None = None,
        multistep_output: int | None = None,
        timestep: str | None = None,
        rollout: dict | None = None,
        validation_rollout: int | None = None,
        **kwargs,
    ) -> None:

        if len(kwargs) > 0:
            LOGGER.warning(
                "The following extra parameters were provided to %s but will be ignored: %s",
                self.__class__.__name__,
                kwargs,
            )

        if multistep_input is not None or multistep_output is not None or timestep is not None:
            assert not input_offsets and not output_offsets and rollout_shift == "0H", (
                "When using multistep_input, multistep_output, and timestep, input_offsets, "
                " output_offsets and rollout_shift must not be provided."
            )
            timestep = frequency_to_timedelta(timestep)
            input_offsets = sorted(-i * timestep for i in range(multistep_input))
            output_offsets = sorted((i + 1) * timestep for i in range(multistep_output))
            rollout_shift = timestep * multistep_output
        else:
            assert multistep_input is None and multistep_output is None and timestep is None, (
                "When using input_offsets, output_offsets, and rollout_shift, multistep_input, "
                "multistep_output, and timestep must not be provided."
            )
            input_offsets = sorted(frequency_to_timedelta(v) for v in input_offsets)
            output_offsets = sorted(frequency_to_timedelta(v) for v in output_offsets)
            rollout_shift = frequency_to_timedelta(rollout_shift)

        super().__init__(input_offsets=input_offsets, output_offsets=output_offsets)
        self._rollout_shift = rollout_shift

        if consistency_check:
            self._validate_offsets()
            self._validate_rollout_shift()

        self.rollout = RolloutConfig(**(rollout or {}))
        self.validation_rollout = validation_rollout
        self._advance_map = self._compute_advance_map()
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

    def _compute_advance_map(self) -> dict[str, list[tuple[int, int]]]:
        """Pre-compute index mappings for input advancement during a rollout step."""
        out_to_idx = {o: j for j, o in enumerate(self._output_offsets)}
        in_to_idx = {i: j for j, i in enumerate(self._input_offsets)}
        advance_map = {"inin": [], "outin": []}
        for new_idx, in_offset in enumerate(self._input_offsets):
            shifted_in = in_offset + self._rollout_shift
            if shifted_in in out_to_idx:
                advance_map["outin"].append((out_to_idx[shifted_in], new_idx))
            else:
                advance_map["inin"].append((in_to_idx[shifted_in], new_idx))
        return advance_map

    def _compute_rollout_offsets(self, rollout_step: int) -> list[datetime.timedelta]:
        """Compute the full list of offsets needed for the current rollout configuration."""
        all_offsets = set(self._input_offsets)
        for step in range(rollout_step):
            shift = self._rollout_shift * step
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
        shift = self._rollout_shift * rollout_step
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
        # Return a fresh tensor: gradient computations need the version of x at each rollout step
        x = x.clone()

        # Shift part of input to be reused.
        for old_idx, new_idx in self._advance_map["inin"]:
            x[:, new_idx] = x[:, old_idx]

        # Compute batch indices for the output offsets of this rollout step
        output_batch_indices = self.get_batch_output_indices(rollout_step=rollout_step)

        for out_idx, new_idx in self._advance_map["outin"]:
            # Get prognostic variables
            x[:, new_idx, ..., data_indices.model.input.prognostic] = y_pred[
                :,
                out_idx,
                ...,
                data_indices.model.output.prognostic,
            ]

            batch_time_index = output_batch_indices[out_idx]
            true_state = batch[:, batch_time_index]

            if output_mask is not None and true_state.shape[1] == 1 and x[:, new_idx].shape[1] != 1:
                true_state = true_state.expand(-1, x[:, new_idx].shape[1], -1, -1)

            x[:, new_idx] = output_mask.rollout_boundary(
                x[:, new_idx],
                true_state,
                data_indices,
                grid_shard_slice=grid_shard_slice,
            )

            # get new "constants" needed for time-varying fields
            x[:, new_idx, ..., data_indices.model.input.forcing] = batch[
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
        offsets = self._offsets
        timestep = min(offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1))
        return frequency_to_string(timestep)

    def _validate_offsets(self) -> None:
        """Check that input and output offsets are well-formed for a forecasting task."""
        if len(self._input_offsets) != len(set(self._input_offsets)):
            msg = f"input_offsets contains duplicate values: {[frequency_to_string(v) for v in self._input_offsets]}"
            raise ValueError(msg)
        if len(self._output_offsets) != len(set(self._output_offsets)):
            msg = f"output_offsets contains duplicate values: {[frequency_to_string(v) for v in self._output_offsets]}"
            raise ValueError(msg)
        if max(self._input_offsets) >= min(self._output_offsets):
            msg = (
                "All output offsets must be strictly greater than all input offsets "
                "for a forecasting task. "
                f"input_offsets={[frequency_to_string(v) for v in self._input_offsets]}, "
                f"output_offsets={[frequency_to_string(v) for v in self._output_offsets]}"
            )
            raise ValueError(msg)

    def _validate_rollout_shift(self) -> None:
        """Check if the rollout shift is valid or replace 0 by the maximum valid shift.

        A shift S is valid if it is strictly positive, the shifted input offsets
        are contained in the union of input and output offsets, and no pairwise
        difference of output offsets is a multiple of S (which would cause the
        same output time step to be forecasted more than once across rollout steps).
        """
        max_input = max(self._input_offsets)
        candidates = [o - max_input for o in self._output_offsets]
        output_diffs = [o2 - o1 for o1 in self._output_offsets for o2 in self._output_offsets if o2 > o1]
        valid = [
            s
            for s in candidates
            if all(i + s in self._offsets for i in self._input_offsets[:-1])
            and all(diff % s != datetime.timedelta(0) for diff in output_diffs)
        ]

        if self._rollout_shift == frequency_to_timedelta("0H"):
            if not valid:
                msg = (
                    "No valid autoregressive rollout shift exists. "
                    "This forecaster cannot be trained with rollout, "
                    "nor can it predict autoregressively in inference. "
                    "If you insist on training a forecaster with these offsets you can set "
                    "`consistency_check=False` in the task configuration.\n"
                    f"input_offsets={[frequency_to_string(v) for v in self._input_offsets]}, "
                    f"output_offsets={[frequency_to_string(v) for v in self._output_offsets]}"
                )
                raise ValueError(msg)
            LOGGER.info("Inferred rollout_shift=%s (maximum valid shift).", frequency_to_string(valid[-1]))
            self._rollout_shift = valid[-1]

        elif self._rollout_shift not in valid:
            msg = (
                f"rollout_shift={frequency_to_string(self._rollout_shift)!r} is not a valid autoregressive "
                "rollout shift for the chosen input and output offsets.\n "
                f"(valid shifts are: {[frequency_to_string(v) for v in valid]}). "
                f"input_offsets={[frequency_to_string(v) for v in self._input_offsets]}, "
                f"output_offsets={[frequency_to_string(v) for v in self._output_offsets]}"
            )
            raise ValueError(msg)
