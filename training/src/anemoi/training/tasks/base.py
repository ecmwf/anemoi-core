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
from abc import ABC
from collections.abc import Iterable
from collections.abc import Mapping
from typing import TYPE_CHECKING

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.utils.time_indices import normalize_time_indices
from anemoi.utils.dates import frequency_to_timedelta

if TYPE_CHECKING:
    from anemoi.models.data import BaseTemplate
    from anemoi.models.data import Batch

LOGGER = logging.getLogger(__name__)


class BaseTask(ABC):
    """Base class for all tasks.

    Tasks define the temporal structure of a training sample by specifying
    input and output time offsets as ``timedelta`` objects. The base class
    provides:

    * An ``_offsets`` property that is the union of input and output offsets,
      used by the datamodule to determine which time steps to load.
    * ``get_inputs`` / ``get_targets`` to split a loaded batch into model
      inputs and targets based on position within ``_offsets``.

    Attributes
    ----------
    name : str
        Name of the task.
    num_inputs : int
        Number of input time steps for the task.
    num_outputs : int
        Number of output time steps for the task.

    """

    name: str

    def __init__(
        self,
        input_offsets: list[datetime.timedelta],
        output_offsets: list[datetime.timedelta],
    ) -> None:
        self.dataset_input_relative_times_by_dataset: dict[str, list[int]] = {}
        self.dataset_target_relative_times_by_dataset: dict[str, list[int]] = {}
        self.dataset_target_relative_times_by_dataset_by_step: dict[str, list[list[int]]] = {}
        self.dataset_time_maps: dict[str, dict[int, int]] = {}
        self.num_input_timesteps_by_dataset: dict[str, int] = {}
        self.num_output_timesteps_by_dataset: dict[str, int] = {}
        self.model_timestep: datetime.timedelta | None = None
        self._input_offsets = sorted(input_offsets)
        self._output_offsets = sorted(output_offsets)
        self._offsets = sorted(set(self._input_offsets + self._output_offsets))

    def steps(self, mode: str = "training") -> Iterable[dict]:  # noqa: ARG002
        """Get the steps for the task."""
        return ({},)  # default is a single step with no kwargs

    @property
    def num_input_timesteps(self) -> int:
        """Number of input time steps."""
        return len(self._input_offsets)

    @property
    def num_output_timesteps(self) -> int:
        """Number of output time steps."""
        return len(self._output_offsets)

    @property
    def num_steps(self) -> int:
        """Number of training steps (rollout length)."""
        return len(self.steps("training"))

    def get_metric_name(self, **_step_kwargs) -> str:
        """Get the metric name for the current step (if any)."""
        return ""

    def get_input_offsets(self, **_kwargs) -> list[datetime.timedelta]:
        """Get the list of input time offsets."""
        return self._input_offsets

    def get_output_offsets(self, **_kwargs) -> list[datetime.timedelta]:
        """Return the output offsets for a given step.

        The default implementation returns ``self._output_offsets``.
        Subclasses may override this to shift outputs per rollout step.
        """
        return self._output_offsets

    def get_offsets(self, **_kwargs) -> list[datetime.timedelta]:
        """Get the list of offsets for a given mode (e.g. "training", "validation", "test").

        By default, this returns ``self._offsets``, but can be overridden by subclasses to return
        different offsets per mode for example (e.g different rollout in training vs validation).
        """
        return self._offsets

    def _offsets_to_batch_indices(self, offsets: list[datetime.timedelta], **kwargs) -> list[int]:
        """Map a list of offsets to their positions in ``self._offsets``."""
        full = self.get_offsets(**kwargs)
        return [full.index(o) for o in offsets]

    def get_batch_input_indices(self, **kwargs) -> list[int]:
        """Positions of the input offsets within the full batch ``_offsets``."""
        return self._offsets_to_batch_indices(self.get_input_offsets(**kwargs))

    def get_batch_output_indices(self, **kwargs) -> list[int]:
        """Positions of the output offsets within the full batch ``_offsets``.

        Parameters are forwarded to ``get_output_offset`` so that
        subclasses can parametrise the output selection (e.g. per
        rollout step).
        """
        return self._offsets_to_batch_indices(self.get_output_offsets(**kwargs))

    def _requested_input_relative_times(self, dataset_name: str, **_kwargs) -> list[int]:
        requested = self.dataset_input_relative_times_by_dataset.get(dataset_name)
        if requested is not None:
            return requested
        return self.get_batch_input_indices()

    def _requested_output_relative_times(self, dataset_name: str, **kwargs) -> list[int]:
        requested_by_step = self.dataset_target_relative_times_by_dataset_by_step.get(dataset_name)
        if requested_by_step is not None:
            step = int(kwargs.get("rollout_step", kwargs.get("step", 0)))
            if not 0 <= step < len(requested_by_step):
                msg = f"Dataset '{dataset_name}' has no mixed-frequency targets for rollout step {step}."
                raise ValueError(msg)
            return requested_by_step[step]
        requested = self.dataset_target_relative_times_by_dataset.get(dataset_name)
        if requested is not None:
            return requested
        return self.get_batch_output_indices(**kwargs)

    def _sample_batch_position(self, dataset_name: str, relative_time: int) -> int:
        time_map = self.dataset_time_maps[dataset_name]
        if relative_time in time_map:
            return time_map[relative_time]
        previous = [value for value in time_map if value <= relative_time]
        if not previous:
            msg = (
                f"Dataset '{dataset_name}' has no timestamp at or before relative time {relative_time}. "
                f"Available times: {sorted(time_map)}"
            )
            raise ValueError(msg)
        return time_map[max(previous)]

    def get_inputs(
        self,
        batch: "Batch",
        data_indices: dict[str, IndexCollection],
        **_kwargs,
    ) -> "Batch":
        """Extract model inputs from a Batch, preserving coords and metadata.

        Parameters
        ----------
        batch : Batch
            Full batch object (data, coords, metadata) keyed by dataset name.
        data_indices : dict[str, IndexCollection]
            Data indices per dataset.

        Returns
        -------
        Batch
            New Batch with input tensors per dataset (shape ``(bs, num_inputs, grid, nvar)``),
            sharing coords and metadata by reference.
        """
        var_indices = {dataset_name: data_indices[dataset_name].data.input.full for dataset_name in batch.dataset_names}
        new_batch = self.get_input_times(batch, **_kwargs).select(variables=var_indices)
        for dataset_name, selected_source in new_batch.items():
            LOGGER.debug("Selected inputs: x[%s] = %s", dataset_name, selected_source)
        return new_batch

    def get_input_times(self, batch: "Batch", **kwargs) -> "Batch":
        """Select the time windows used as model inputs, retaining all variables."""
        if self.dataset_time_maps:
            time_indices = {
                name: normalize_time_indices(
                    [
                        self._sample_batch_position(name, value)
                        for value in self._requested_input_relative_times(name, **kwargs)
                    ],
                )
                for name in batch.dataset_names
            }
        else:
            time_indices = normalize_time_indices(self.get_batch_input_indices())
        return batch.select(time=time_indices)

    def get_targets(
        self,
        batch: "Batch",
        data_indices: dict[str, IndexCollection],
        **kwargs,
    ) -> tuple["Batch", dict[str, "BaseTemplate"], "Batch"]:
        """Extract model targets from a Batch, preserving coords and metadata.

        Parameters
        ----------
        batch : Batch
            Full batch object (data, coords, metadata) keyed by dataset name.
        data_indices : dict[str, IndexCollection]
            Data indices per dataset.
        **kwargs
            Forwarded to ``get_batch_output_indices`` (e.g. ``rollout_step``).

        Returns
        -------
        tuple[Batch, dict[str, BaseTemplate], Batch]: (target, target_template, target_forcing)
            target holds the target tensors per dataset, shape (bs, num_outputs, ensemble, grid, full_nvar)).
            target_template describes what the model predicts at the target nodes: the model's output
            variables, in the model's output order, with their statistics, and no data.
            target_forcing contains the output-time forcing variables used to
            condition the decoder, preserving coordinates, timedeltas,
            metadata and layouts.
        """
        if self.dataset_time_maps:
            time_indices = {
                name: [
                    self._sample_batch_position(name, value)
                    for value in self._requested_output_relative_times(name, **kwargs)
                ]
                for name in batch.dataset_names
            }
        else:
            time_indices = dict.fromkeys(batch.dataset_names, self.get_batch_output_indices(**kwargs))
        for dataset_name, view in batch.items():
            dataset_indices = time_indices[dataset_name]
            if dataset_indices and max(dataset_indices) >= view.time_size:
                msg = (
                    f"Batch for dataset '{dataset_name}' contains {view.time_size} time steps, "
                    f"but requires index {max(dataset_indices)} (indices {dataset_indices}). "
                    "The dataloader's time window does not match the task rollout."
                )
                raise ValueError(msg)

        time_indices = {name: normalize_time_indices(indices) for name, indices in time_indices.items()}

        target_tensors = self.measure_targets_from_step(batch.select(time=time_indices), **kwargs)
        for dataset_name, selected_source in target_tensors.items():
            LOGGER.debug("Selected targets: x[%s] = %s", dataset_name, selected_source)

        var_indices = {
            dataset_name: data_indices[dataset_name].data.input.forcing for dataset_name in batch.dataset_names
        }
        target_forcing = target_tensors.select(variables=var_indices)

        return target_tensors, self.get_target_template(target_tensors, data_indices), target_forcing

    def measure_targets_from_step(self, targets: "Batch", **_kwargs) -> "Batch":
        """Return ``targets`` with their point times measured from the step's own reference time.

        Batches measure point times from the sample's reference time. Tasks whose steps move
        that reference (e.g. rollout) shift the targets' times here.
        """
        return targets

    @staticmethod
    def get_target_template(
        targets: "Batch",
        data_indices: dict[str, IndexCollection],
    ) -> dict[str, "BaseTemplate"]:
        """Return what the model predicts at the nodes of ``targets``, one template per dataset.

        The targets carry every variable of a dataset; each template keeps the model's output
        variables, in the model's output order, together with their statistics.
        """
        templates = {}
        for dataset_name, source in targets.items():
            output_names = data_indices[dataset_name].model.output.ordered_names
            positions = [source.name_to_index[name] for name in output_names]
            templates[dataset_name] = source.template().select_variables(positions)
        return templates

    def log_extra(self, *_args, **_kwargs) -> None:  # noqa: B027
        """Hook to log any task-specific information."""

    def log_training_state(self) -> None:  # noqa: B027
        """Log the effective task state at the start of training."""

    def training_runtime_state_dict(self) -> dict:
        """Return training runtime state to be persisted in the training checkpoint.

        Override in subclasses to include any mutable state that accumulates
        during training and must survive a job resume (e.g. curriculum counters).
        This state is stored at the training checkpoint level, not in the model
        state_dict, so that it is invisible to inference code.
        """
        return {}

    def load_training_runtime_state_dict(self, state: dict) -> None:  # noqa: B027
        """Restore training runtime state from a training checkpoint."""

    def on_train_epoch_end(self, current_epoch: int) -> None:  # noqa: B027
        """Hook to update task state at the end of each training epoch (e.g. for curriculum learning)."""

    def fill_metadata(self, md_dict: dict) -> None:
        """Fill the metadata dictionary with task-specific information."""
        md_dict["metadata_inference"]["task"] = self.name

        input_relative_date_indices = self.get_batch_input_indices()
        output_relative_date_indices = self.get_batch_output_indices()
        timestep = self._get_timestep_for_metadata()
        relative_date_indices = sorted(input_relative_date_indices + output_relative_date_indices)
        timesteps = {
            "relative_date_indices_training": relative_date_indices,  # backwards compatibility with inference
            "input_relative_date_indices": input_relative_date_indices,  # backwards compatibility with inference
            "output_relative_date_indices": output_relative_date_indices,  # backwards compatibility with inference
            "timestep": timestep,  # backwards compatibility with inference
        }

        dataset_names = md_dict["metadata_inference"]["dataset_names"]
        for dataset_name in dataset_names:
            existing = md_dict["metadata_inference"][dataset_name].get("timesteps", {})
            dataset_timesteps = timesteps | existing
            inputs = existing.get("relative_date_input_indices_training_by_dataset", {})
            relative = existing.get("relative_date_indices_training_by_dataset", {})
            targets = existing.get("relative_date_target_indices_training_by_dataset_by_step", {})
            if dataset_name in inputs:
                dataset_timesteps["input_relative_date_indices"] = inputs[dataset_name]
            if dataset_name in relative:
                dataset_timesteps["relative_date_indices_training"] = relative[dataset_name]
            if targets.get(dataset_name):
                dataset_timesteps["output_relative_date_indices"] = targets[dataset_name][0]
            if "model_timestep" in existing:
                dataset_timesteps["timestep"] = existing["model_timestep"]
            md_dict["metadata_inference"][dataset_name]["timesteps"] = dataset_timesteps

    @staticmethod
    def _relative_times(metadata: Mapping, dataset_names: list[str], keys: tuple[str, ...]) -> dict[str, list[int]]:
        result = {}
        for name in dataset_names:
            timesteps = metadata.get(name, {}).get("timesteps", {})
            chosen = None
            for key in keys:
                values = timesteps.get(key)
                if isinstance(values, Mapping) and name in values:
                    candidate = [int(value) for value in values[name]]
                    if chosen is None or max(candidate, default=-1) > max(chosen, default=-1):
                        chosen = candidate
            if chosen is not None:
                result[name] = chosen
        return result

    @staticmethod
    def _relative_times_by_step(
        metadata: Mapping,
        dataset_names: list[str],
        keys: tuple[str, ...],
    ) -> dict[str, list[list[int]]]:
        result = {}
        for name in dataset_names:
            timesteps = metadata.get(name, {}).get("timesteps", {})
            chosen = None
            for key in keys:
                values = timesteps.get(key)
                if isinstance(values, Mapping) and name in values:
                    candidate = [[int(value) for value in step] for step in values[name]]
                    candidate_max = max((value for step in candidate for value in step), default=-1)
                    chosen_max = max((value for step in (chosen or []) for value in step), default=-1)
                    if chosen is None or candidate_max > chosen_max:
                        chosen = candidate
            if chosen is not None:
                result[name] = chosen
        return result

    def configure_from_metadata(self, md_dict: dict) -> None:
        """Configure dataset-specific time selection from datamodule metadata."""
        metadata = md_dict.get("metadata_inference", {})
        dataset_names = metadata.get("dataset_names", []) if isinstance(metadata, Mapping) else []
        self.dataset_input_relative_times_by_dataset = {}
        self.dataset_target_relative_times_by_dataset = {}
        self.dataset_target_relative_times_by_dataset_by_step = {}
        self.dataset_time_maps = {}
        self.num_input_timesteps_by_dataset = {}
        self.num_output_timesteps_by_dataset = {}
        self.model_timestep = None
        if not dataset_names:
            return

        first_timesteps = metadata.get(dataset_names[0], {}).get("timesteps", {})
        if "model_timestep" not in first_timesteps:
            return
        self.model_timestep = frequency_to_timedelta(first_timesteps["model_timestep"])
        self.dataset_input_relative_times_by_dataset = self._relative_times(
            metadata,
            dataset_names,
            ("relative_date_input_indices_validation_by_dataset", "relative_date_input_indices_training_by_dataset"),
        )
        self.dataset_target_relative_times_by_dataset = self._relative_times(
            metadata,
            dataset_names,
            ("relative_date_target_indices_validation_by_dataset", "relative_date_target_indices_training_by_dataset"),
        )
        self.dataset_target_relative_times_by_dataset_by_step = self._relative_times_by_step(
            metadata,
            dataset_names,
            (
                "relative_date_target_indices_validation_by_dataset_by_step",
                "relative_date_target_indices_training_by_dataset_by_step",
            ),
        )
        self.num_input_timesteps_by_dataset = {
            name: len(self._requested_input_relative_times(name)) for name in dataset_names
        }
        for name in dataset_names:
            by_step = self.dataset_target_relative_times_by_dataset_by_step.get(name)
            counts = (
                {len(values) for values in by_step}
                if by_step is not None
                else {len(self._requested_output_relative_times(name))}
            )
            if len(counts) != 1:
                msg = f"Dataset '{name}' has a different number of native targets per rollout step."
                raise ValueError(msg)
            self.num_output_timesteps_by_dataset[name] = counts.pop()

        relative = self._relative_times(
            metadata,
            dataset_names,
            ("relative_date_indices_validation_by_dataset", "relative_date_indices_training_by_dataset"),
        )
        self.dataset_time_maps = {
            name: {value: position for position, value in enumerate(relative[name])} for name in dataset_names
        }


class BaseSingleStepTask(BaseTask):
    """Base class for single-step tasks."""

    def advance_input(self, batch: "Batch", *_args, **_kwargs) -> "Batch":
        """Advance the input state for each dataset based on the task's requirements.

        This method can be overridden by specific tasks to implement custom logic for advancing the input state.

        Parameters
        ----------
        batch : Batch
            The input state for each dataset.
        *_args
            Additional positional arguments, which are ignored by the default implementation.
        **_kwargs
            Additional keyword arguments, which are ignored by the default implementation.

        Returns
        -------
        Batch
            The advanced input state for each dataset (default: passthrough).
        """
        return batch
