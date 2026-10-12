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
from typing import TYPE_CHECKING

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.diagnostics.callbacks.plot_adapter import ForecasterPlotAdapter
from anemoi.training.tasks.base import BaseTask
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from anemoi.models.data.batch import Batch
    from anemoi.models.data.sources import Source
    from anemoi.models.data.sources import TabularSource


class RolloutConfig:
    """Rollout configuration for autoregressive training."""

    def __init__(self, start: int = 1, epoch_increment: int = 0, maximum: int = 1) -> None:
        """Initialize rollout configuration."""
        self.start = start
        self.epoch_increment = epoch_increment
        self.maximum = maximum
        self.step = self.start
        # Remember which epoch last increased the rollout so that running the
        # hook again for the same epoch does not increase it twice.
        self._last_increased_epoch: int = -1

    def should_increase(self, current_epoch: int) -> bool:
        """Check if rollout should be increased at the end of the current epoch."""
        return (
            self.epoch_increment > 0
            and (current_epoch + 1) % self.epoch_increment == 0
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


class BaseForecaster(BaseTask):
    """Abstract forecasting task implementation.

    For rollout training, training offsets extend up to the current
    ``rollout.step`` so the dataloader only loads the required time
    steps. ``rollout.step`` grows via ``on_train_epoch_end``.
    """

    name: str

    def __init__(
        self,
        input_offsets: list[datetime.timedelta],
        output_offsets: list[datetime.timedelta],
        rollout_shift: datetime.timedelta,
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

        super().__init__(input_offsets=input_offsets, output_offsets=output_offsets)

        self._rollout_shift = rollout_shift
        self.rollout = RolloutConfig(**(rollout or {}))
        self.validation_rollout = validation_rollout
        self._advance_map = self._compute_advance_map()
        self._plot_adapter = ForecasterPlotAdapter(self)

    def steps(self, mode: str = "training") -> tuple[dict[str, int], ...]:
        """Return the current steps configuration based on the rollout step."""
        max_rollout = self.rollout.step
        if mode == "validation" and self.validation_rollout is not None:
            max_rollout = max(max_rollout, self.validation_rollout)
        return tuple({"rollout_step": i} for i in range(max_rollout))

    def get_metric_name(self, rollout_step: int = 0, **_kwargs) -> str:
        """Get the metric name for the current step."""
        return f"_rstep{rollout_step}"

    def _compute_rollout_offsets(self, rollout_step: int) -> list[datetime.timedelta]:
        """Compute the full list of offsets needed for the current rollout configuration."""
        all_offsets = set(self._input_offsets)
        for step in range(rollout_step):
            shift = self._rollout_shift * step
            for o in self._output_offsets:
                all_offsets.add(o + shift)
        return sorted(all_offsets)

    def get_offsets(self, mode: str | None = None) -> list[datetime.timedelta]:
        if mode in ("training", "validation"):
            rollout_step = len(self.steps(mode))
        else:
            LOGGER.debug(
                "Unknown mode '%s' for %s.get_offsets(); using offsets for the longest configured rollout.",
                mode,
                self.__class__.__name__,
            )
            rollout_step = max(self.rollout.maximum, self.validation_rollout or 0)

        return self._compute_rollout_offsets(rollout_step)

    def get_output_offsets(
        self,
        rollout_step: int = 0,
        **_kwargs,
    ) -> list[datetime.timedelta]:
        """Return output offsets shifted by ``rollout_step``."""
        shift = self._rollout_shift * rollout_step
        return sorted(o + shift for o in self._output_offsets)

    def measure_targets_from_step(self, targets: "Batch", rollout_step: int = 0, **_kwargs) -> "Batch":
        """Measure the targets' point times from the forecast time of ``rollout_step``.

        Each rollout step moves the forecast time on by one rollout shift.
        """
        return _shift_point_times(targets, -self._rollout_shift * rollout_step)

    def _advance_gridded_input(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor | None,
        output_values: torch.Tensor,
        data_indices: IndexCollection | None = None,
        output_mask: object | None = None,
        grid_shard_slice: slice | None = None,
    ) -> torch.Tensor:
        """Advance a gridded dataset's input state for the next rollout step.

        Input windows that remain inputs move to their new position (see ``_compute_advance_map``).
        Each new input window is the output window at the same time: the predicted prognostics,
        the true state outside the output mask, and the output-time forcings. A dataset without
        a prediction (it has no decoder) takes its true state there, forcings included.
        Supports model outputs shaped like ``(B, T, E, G, V)``.
        """
        # Return a fresh tensor: gradient computations need the version of x at each rollout step
        previous = x
        x = x.clone()

        # Shift part of input to be reused.
        for old_idx, new_idx in self._advance_map["inin"]:
            x[:, new_idx] = previous[:, old_idx]

        for out_idx, new_idx in self._advance_map["outin"]:
            if y_pred is None:
                # No prediction (the dataset has no decoder): its next input is its true state,
                # forcings included.
                x[:, new_idx] = output_values[:, out_idx, ..., data_indices.data.input.full]
                continue

            # Get prognostic variables
            x[:, new_idx, ..., data_indices.model.input.prognostic] = y_pred[
                :,
                out_idx,
                ...,
                data_indices.model.output.prognostic,
            ]

            true_state = output_values[:, out_idx]

            if output_mask is not None and true_state.shape[1] == 1 and x[:, new_idx].shape[1] != 1:
                true_state = true_state.expand(-1, x[:, new_idx].shape[1], -1, -1)

            x[:, new_idx] = output_mask.rollout_boundary(
                x[:, new_idx],
                true_state,
                data_indices,
                grid_shard_slice=grid_shard_slice,
            )

            # get new "constants" needed for time-varying fields
            x[:, new_idx, ..., data_indices.model.input.forcing] = output_values[
                :,
                out_idx,
                ...,
                data_indices.data.input.forcing,
            ]
        return x

    def _compute_advance_map(self) -> dict[str, list[tuple[int, int]]]:
        """Map each input window of the next rollout step to where it comes from.

        Returns ``{"inin": [(old_input_index, new_input_index), ...], "outin": [(output_index,
        new_input_index), ...]}``: after a step the input window at ``input_offsets[new]`` is
        either a current input window (shifted by ``rollout_shift``) or an output window.
        """
        out_to_idx = {o: j for j, o in enumerate(self._output_offsets)}
        in_to_idx = {i: j for j, i in enumerate(self._input_offsets)}
        advance_map = {"inin": [], "outin": []}
        for new_idx, in_offset in enumerate(self._input_offsets):
            shifted_in = in_offset + self._rollout_shift
            if shifted_in in out_to_idx:
                advance_map["outin"].append((out_to_idx[shifted_in], new_idx))
            elif shifted_in in in_to_idx:
                advance_map["inin"].append((in_to_idx[shifted_in], new_idx))
            else:
                msg = (
                    f"Cannot advance the input window at offset {in_offset} by the rollout shift "
                    f"{self._rollout_shift}: {shifted_in} is neither an input nor an output offset."
                )
                raise ValueError(msg)
        return advance_map

    def advance_input(
        self,
        x: "Batch",
        y_pred: "Batch",
        output_values: "Batch",
        rollout_step: int = 0,
        data_indices: dict[str, IndexCollection] | None = None,
        output_mask: dict[str, object] | None = None,
        grid_shard_slice: dict[str, slice | None] | None = None,
    ) -> "Batch":
        """Advance the input state for the next rollout step, preserving coords and metadata.

        A dataset without a prediction (``None``: it has no decoder) takes its true state at the
        next input times. Tabular (observation) datasets advance their time windows the same way
        (see ``_advance_tabular_input``).
        """
        del rollout_step
        return x.with_sources(
            {
                dataset_name: self._advance_source(
                    view,
                    y_pred.get(dataset_name),
                    output_values[dataset_name],
                    data_indices=data_indices[dataset_name],
                    output_mask=None if output_mask is None else output_mask[dataset_name],
                    grid_shard_slice=None if grid_shard_slice is None else grid_shard_slice.get(dataset_name),
                )
                for dataset_name, view in x.items()
            },
        )

    def _advance_source(
        self,
        x: "Source",
        prediction: "Source | None",
        truth: "Source",
        data_indices: IndexCollection,
        output_mask: object | None = None,
        grid_shard_slice: slice | None = None,
    ) -> "Source":
        """Advance one dataset's input for the next rollout step.

        Parameters
        ----------
        x : Source
            Current input, in the model input variables.
        prediction : Source | None
            Predicted output, in the model output variables, or ``None`` if the dataset is not decoded.
        truth : Source
            True output, in the full data variables.
        data_indices : IndexCollection
            Data indices of the dataset.
        output_mask : object | None
            Output mask of a gridded dataset, which refills the true state outside it.
        grid_shard_slice : slice | None
            Local grid shard of a gridded dataset, which the output mask indexes.

        Returns
        -------
        Source
            The input for the next rollout step.
        """
        if x.is_tabular:
            return self._advance_tabular_input(x, prediction, truth, data_indices)

        new_data = self._advance_gridded_input(
            x.data,
            None if prediction is None else prediction.data.to(x.dtype),
            truth.data,
            data_indices=data_indices,
            output_mask=output_mask,
            grid_shard_slice=grid_shard_slice,
        )
        return x.clone(data=new_data)

    def _advance_tabular_input(
        self,
        x: "TabularSource",
        prediction: "TabularSource | None",
        truth: "TabularSource",
        data_indices: IndexCollection,
    ) -> "TabularSource":
        """Advance an observation dataset's input windows for the next rollout step.

        Input windows that remain inputs move to their new position. Each new input window is
        the output window at the same time: a dataset the model decodes takes its predicted
        values there for its prognostic variables, with the observed forcings; a dataset it does
        not decode takes the observations. Timedeltas are measured from the current forecast
        time, which moves on by one rollout shift, so every window's timedeltas drop by that shift.

        Parameters
        ----------
        x : TabularSource
            Current input windows, in the model input variables.
        prediction : TabularSource | None
            Predicted output windows, in the model output variables, or ``None`` if the dataset
            is not decoded.
        truth : TabularSource
            Observed output windows, in the full data variables.
        data_indices : IndexCollection
            Data indices of the dataset.

        Returns
        -------
        TabularSource
            The input windows for the next rollout step.
        """
        windows: dict[int, TabularSource] = {}
        advance_map = self._advance_map
        for old_idx, new_idx in advance_map["inin"]:
            windows[new_idx] = x.select_time([old_idx])
        for out_idx, new_idx in advance_map["outin"]:
            window = truth.select_time([out_idx]).select_variables(data_indices.data.input.full)
            window = _match_ensemble_size(window, x.ensemble_size)
            if prediction is not None:
                window = _with_predicted_prognostics(window, prediction.select_time([out_idx]), data_indices)
            windows[new_idx] = window.map_data(lambda t: t.to(x.dtype))

        if sorted(windows) != list(range(x.time_size)):
            msg = (
                f"Source {x.name!r} has {x.time_size} input windows, but the rollout advance map "
                f"{advance_map} fills {sorted(windows)}."
            )
            raise ValueError(msg)
        ordered = [windows[i] for i in range(x.time_size)]
        advanced = ordered[0].concat_time(*ordered[1:])
        shift = self._rollout_shift.total_seconds()
        return advanced.clone(timedeltas=[timedeltas - shift for timedeltas in advanced.timedeltas])

    def log_extra(self, logger: Callable, logger_enabled: bool, batch_size: int | None = None) -> None:
        """Log any task-specific information."""
        logger(
            "rollout",
            float(self.rollout.step),
            on_step=False,
            on_epoch=True,
            logger=logger_enabled,
            rank_zero_only=True,
            sync_dist=False,
            batch_size=batch_size,
        )

    def log_training_state(self) -> None:
        """Log the effective rollout state at the start of training."""
        LOGGER.info("Effective task rollout step: %d.", self.rollout.step)

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
            initialized_step = self.rollout.step
            self.rollout.load_state_dict(state["rollout"])
            LOGGER.info(
                "Restored rollout step from checkpoint: %d (task was initialized at step %d).",
                self.rollout.step,
                initialized_step,
            )

    def on_train_epoch_end(self, current_epoch: int) -> None:
        if self.rollout.should_increase(current_epoch):
            self.rollout.increase(current_epoch)

    def _get_timestep_for_metadata(self) -> str:
        """Get the timestep string for metadata."""
        offsets = self._offsets
        timestep = min(offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1))
        return frequency_to_string(timestep)


class Forecaster(BaseForecaster):
    """Basic Forecasting task implementation.

    Builds input and output offsets from ``multistep_input``,
    ``multistep_output`` and a ``timestep`` string (e.g. ``"6H"``).
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

        # Input: e.g. multistep_input=2, timestep=6H     ->  [-6H, 0H]
        input_offsets = [-1 * i * self.timestep for i in range(multistep_input)]
        # Outputs: e.g. multistep_output=1, timestep=6H  -> [[6H], [12H], [18H], ...] up to rollout.maximum
        output_offsets = [(i + 1) * self.timestep for i in range(multistep_output)]
        rollout_shift = self.timestep * self.num_output_steps

        super().__init__(
            input_offsets=input_offsets,
            output_offsets=output_offsets,
            rollout_shift=rollout_shift,
            rollout=rollout,
            validation_rollout=validation_rollout,
            **kwargs,
        )


class OffsetForecaster(BaseForecaster):
    """Alternative Forecasting task implementation.

    Forecaster directly configured from offsets as lists of strings.
    Offsets are validated for consistency with rollout.
    The default rollout shift is the maximum valid shift.
    """

    name: str = "offset-forecaster"

    def __init__(
        self,
        input_offsets: list[str],
        output_offsets: list[str],
        rollout_shift: str = "default",
        rollout: dict | None = None,
        validation_rollout: int | None = None,
        **kwargs,
    ) -> None:

        input_offsets, output_offsets, rollout_shift = self._convert_and_validate(
            input_offsets,
            output_offsets,
            rollout_shift,
        )

        super().__init__(
            input_offsets=input_offsets,
            output_offsets=output_offsets,
            rollout_shift=rollout_shift,
            rollout=rollout,
            validation_rollout=validation_rollout,
            **kwargs,
        )

    def fill_metadata(self, md_dict: dict) -> None:
        """Fill the metadata dictionary with task-specific information."""
        super().fill_metadata(md_dict)
        fc_timesteps = {
            "input_offsets": [frequency_to_string(o) for o in self._input_offsets],
            "output_offsets": [frequency_to_string(o) for o in self._output_offsets],
            "rollout_shift": frequency_to_string(self._rollout_shift),
            "advance_map": self._advance_map,
        }
        dataset_names = md_dict["metadata_inference"]["dataset_names"]
        for dataset_name in dataset_names:
            md_dict["metadata_inference"][dataset_name]["timesteps"].update(fc_timesteps)

    @staticmethod
    def _convert_and_validate(
        input_offsets: list[str],
        output_offsets: list[str],
        rollout_shift: str,
    ) -> tuple[list[datetime.timedelta], list[datetime.timedelta], datetime.timedelta]:
        """Convert string config to validated timedeltas."""
        input_offsets = sorted(frequency_to_timedelta(v) for v in input_offsets)
        output_offsets = sorted(frequency_to_timedelta(v) for v in output_offsets)

        # Check that input and output offsets are well-formed for a forecasting task.
        if len(input_offsets) != len(set(input_offsets)):
            msg = f"input_offsets contains duplicate values: {[frequency_to_string(v) for v in input_offsets]}"
            raise ValueError(msg)
        if len(output_offsets) != len(set(output_offsets)):
            msg = f"output_offsets contains duplicate values: {[frequency_to_string(v) for v in output_offsets]}"
            raise ValueError(msg)
        if max(input_offsets) != datetime.timedelta(0):
            msg = (
                "The latest input offset must be 0h (the forecast initialisation time). "
                f"input_offsets={[frequency_to_string(v) for v in input_offsets]}"
            )
            raise ValueError(msg)
        if max(input_offsets) >= min(output_offsets):
            msg = (
                "All output offsets must be strictly greater than all input offsets "
                "for a forecasting task. "
                f"input_offsets={[frequency_to_string(v) for v in input_offsets]}, "
                f"output_offsets={[frequency_to_string(v) for v in output_offsets]}"
            )
            raise ValueError(msg)

        # Check if the rollout shift is valid or replace "default" by the maximum valid shift.

        # A shift S is valid if the shifted input offsets are contained in the
        # union of input and output offsets and every output of one rollout step
        # precedes every output of the next step. The latter also prevents the same
        # output time from being forecast more than once across rollout steps.
        max_input = max(input_offsets)
        candidates = [o - max_input for o in output_offsets]
        known_offsets = set(input_offsets + output_offsets)
        output_span = max(output_offsets) - min(output_offsets)
        valid = [s for s in candidates if all(i + s in known_offsets for i in input_offsets[:-1]) and output_span < s]

        if rollout_shift == "default":
            if not valid:
                msg = (
                    "No valid autoregressive rollout shift exists. "
                    "This forecaster cannot be trained with rollout, "
                    "nor can it predict autoregressively in inference.\n"
                    f"input_offsets={[frequency_to_string(v) for v in input_offsets]}, "
                    f"output_offsets={[frequency_to_string(v) for v in output_offsets]}"
                )
                raise ValueError(msg)
            LOGGER.info("Inferred rollout_shift=%s (maximum valid shift).", frequency_to_string(valid[-1]))
            rollout_shift = valid[-1]

        else:
            rollout_shift = frequency_to_timedelta(rollout_shift)
            if rollout_shift not in valid:
                msg = (
                    f"rollout_shift={frequency_to_string(rollout_shift)!r} is not a valid autoregressive "
                    "rollout shift for the chosen input and output offsets.\n "
                    f"(valid shifts are: {[frequency_to_string(v) for v in valid]}). "
                    f"input_offsets={[frequency_to_string(v) for v in input_offsets]}, "
                    f"output_offsets={[frequency_to_string(v) for v in output_offsets]}"
                )
                raise ValueError(msg)
        return input_offsets, output_offsets, rollout_shift


def _shift_point_times(batch: "Batch", shift: datetime.timedelta) -> "Batch":
    """Return ``batch`` with the timedeltas of its tabular sources moved by ``shift``."""
    if not shift:
        return batch
    seconds = shift.total_seconds()
    return batch.with_sources(
        {
            name: (
                source.clone(timedeltas=[timedeltas + seconds for timedeltas in source.timedeltas])
                if source.is_tabular
                else source
            )
            for name, source in batch.items()
        },
    )


def _match_ensemble_size(source: "TabularSource", ensemble_size: int) -> "TabularSource":
    """Broadcast a single-member tabular source to ``ensemble_size`` members."""
    if source.ensemble_size == ensemble_size:
        return source
    if source.ensemble_size != 1:
        msg = (
            f"Cannot use the {source.ensemble_size} members of source {source.name!r} as input for "
            f"{ensemble_size} ensemble members."
        )
        raise ValueError(msg)

    def expand(t: torch.Tensor) -> torch.Tensor:
        axis = source.layout.axis("ensemble", ndim=t.ndim)
        return t.expand(*[ensemble_size if i == axis else -1 for i in range(t.ndim)])

    return source.map_data(expand)


def _with_predicted_prognostics(
    window: "TabularSource",
    prediction: "TabularSource",
    data_indices: IndexCollection,
) -> "TabularSource":
    """Return ``window`` (model input variables) with its prognostic values taken from ``prediction``."""
    if window.layout.axis("variables", ndim=window.layout.ndim) != window.layout.ndim - 1:
        msg = f"Source {window.name!r} must have its variables last to take predicted values, got {window.layout!r}."
        raise ValueError(msg)

    input_prognostic = data_indices.model.input.prognostic
    output_prognostic = data_indices.model.output.prognostic
    new_data = []
    for sample_idx, (observed, predicted) in enumerate(zip(window.data, prediction.data, strict=True)):
        if observed.shape[:-1] != predicted.shape[:-1]:
            msg = (
                f"Source {window.name!r}, sample {sample_idx}: the prediction has shape {tuple(predicted.shape)} "
                f"but the observed window has {tuple(observed.shape)}; the predicted nodes (and ensemble members) "
                "must match the observations of the window they replace."
            )
            raise ValueError(msg)
        sample = observed.clone()
        sample[..., input_prognostic] = predicted[..., output_prognostic].to(sample.dtype)
        new_data.append(sample)
    return window.clone(data=new_data)
