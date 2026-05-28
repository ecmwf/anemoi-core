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

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.utils.time_indices import normalize_time_indices

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

    Methods
    -------
    get_input_offsets() -> list[datetime.timedelta]
        Get the list of input time offsets.
    get_output_offsets(**kwargs) -> list[datetime.timedelta]
        Get the list of output time offsets.
    get_offsets(**kwargs) -> list[datetime.timedelta]
        Get the full list of input and output time offsets.
    """

    name: str

    def __init__(
        self,
        input_offsets: list[datetime.timedelta] | dict[str, list[datetime.timedelta]],
        output_offsets: list[datetime.timedelta] | dict[str, list[datetime.timedelta]],
    ) -> None:
        self.num_input_timesteps_by_dataset: dict[str, int] = {}
        self.num_output_timesteps_by_dataset: dict[str, int] = {}
        self.dataset_input_relative_times_by_dataset: dict[str, list[int]] = {}
        self.dataset_target_relative_times_by_dataset: dict[str, list[int]] = {}
        self.dataset_time_maps: dict[str, dict[int, int]] = {}
        self._reference_input_dataset_name: str | None = None
        self._reference_output_dataset_name: str | None = None

        input_offsets_are_mapping = isinstance(input_offsets, Mapping) or hasattr(input_offsets, "items")
        output_offsets_are_mapping = isinstance(output_offsets, Mapping) or hasattr(output_offsets, "items")

        if input_offsets_are_mapping or output_offsets_are_mapping:
            if not input_offsets_are_mapping or not output_offsets_are_mapping:
                msg = "`input_offsets` and `output_offsets` must both be mappings for per-dataset offsets."
                raise ValueError(msg)
            input_offsets = dict(input_offsets)
            output_offsets = dict(output_offsets)
            dataset_names = sorted(set(input_offsets) | set(output_offsets))
            self._input_offsets = {
                dataset_name: sorted(input_offsets.get(dataset_name, [])) for dataset_name in dataset_names
            }
            self._output_offsets = {
                dataset_name: sorted(output_offsets.get(dataset_name, [])) for dataset_name in dataset_names
            }
            self._offsets = sorted(
                {
                    *[offset for offsets in self._input_offsets.values() for offset in offsets],
                    *[offset for offsets in self._output_offsets.values() for offset in offsets],
                },
            )
            self.num_input_timesteps_by_dataset = {
                dataset_name: len(self._input_offsets[dataset_name]) for dataset_name in dataset_names
            }
            self.num_output_timesteps_by_dataset = {
                dataset_name: len(self._output_offsets[dataset_name]) for dataset_name in dataset_names
            }
            if len(self._input_offsets) > 0:
                self._reference_input_dataset_name = max(
                    self._input_offsets,
                    key=lambda name: len(self._input_offsets[name]),
                )
            if len(self._output_offsets) > 0:
                self._reference_output_dataset_name = max(
                    self._output_offsets,
                    key=lambda name: len(self._output_offsets[name]),
                )
            else:
                self._reference_output_dataset_name = self._reference_input_dataset_name
            return

        self._input_offsets = sorted(input_offsets)
        self._output_offsets = sorted(output_offsets)
        self._offsets = sorted(set(self._input_offsets + self._output_offsets))

    def steps(self, mode: str = "training") -> Iterable[dict]:  # noqa: ARG002
        """Get the steps for the task."""
        return ({},)  # default is a single step with no kwargs

    @property
    def num_input_timesteps(self) -> int:
        """Number of input time steps."""
        return len(self.get_input_offsets())

    @property
    def num_output_timesteps(self) -> int:
        """Number of output time steps."""
        return len(self.get_output_offsets())

    @property
    def num_steps(self) -> int:
        """Number of training steps (rollout length)."""
        return len(self.steps("training"))

    def get_metric_name(self, **_step_kwargs) -> str:
        """Get the metric name for the current step (if any)."""
        return ""

    def get_input_offsets(self, dataset_name: str | None = None, **_kwargs) -> list[datetime.timedelta]:
        """Get the list of input time offsets."""
        if isinstance(self._input_offsets, Mapping):
            if dataset_name is not None and dataset_name in self._input_offsets:
                return self._input_offsets[dataset_name]
            if self._reference_input_dataset_name is not None:
                return self._input_offsets[self._reference_input_dataset_name]
        return self._input_offsets

    def get_output_offsets(self, dataset_name: str | None = None, **_kwargs) -> list[datetime.timedelta]:
        """Return the output offsets for a given step.

        The default implementation returns ``self._output_offsets``.
        Subclasses may override this to shift outputs per rollout step.
        """
        if isinstance(self._output_offsets, Mapping):
            if dataset_name is not None and dataset_name in self._output_offsets:
                return self._output_offsets[dataset_name]
            if self._reference_output_dataset_name is not None:
                return self._output_offsets[self._reference_output_dataset_name]
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

    def _requested_input_relative_times(self, dataset_name: str) -> list[int]:
        requested = self.dataset_input_relative_times_by_dataset.get(dataset_name)
        if requested is not None:
            return requested
        return self.get_batch_input_indices(dataset_name=dataset_name)

    def _requested_output_relative_times(self, dataset_name: str, **kwargs) -> list[int]:
        requested = self.dataset_target_relative_times_by_dataset.get(dataset_name)
        if requested is not None:
            return requested
        return self.get_batch_output_indices(dataset_name=dataset_name, **kwargs)

    def _resolve_relative_time_metadata(
        self,
        metadata_inference: Mapping,
        dataset_names: list[str],
        keys: tuple[str, ...],
    ) -> dict[str, list[int]]:
        """Choose the richest per-dataset time window exposed by the datamodule metadata."""
        relative_by_dataset: dict[str, list[int]] = {}

        for dataset_name in dataset_names:
            dataset_meta = metadata_inference.get(dataset_name, {})
            timesteps_meta = dataset_meta.get("timesteps", {}) if isinstance(dataset_meta, Mapping) else {}

            chosen: list[int] | None = None
            for key in keys:
                raw_relative = timesteps_meta.get(key, None)
                if not isinstance(raw_relative, Mapping):
                    continue
                raw_values = raw_relative.get(dataset_name, None)
                if raw_values is None:
                    continue
                candidate = [int(value) for value in raw_values]
                if chosen is None or max(candidate, default=-1) > max(chosen, default=-1):
                    chosen = candidate

            if chosen is not None:
                relative_by_dataset[dataset_name] = chosen

        return relative_by_dataset

    def _sample_batch_position(self, *, dataset_name: str, relative_time: int) -> int:
        time_map = self.dataset_time_maps.get(dataset_name, {})
        exact_idx = time_map.get(int(relative_time), None)
        if exact_idx is not None:
            return int(exact_idx)

        available_times = sorted(int(value) for value in time_map)
        if not available_times:
            msg = f"Dataset '{dataset_name}' has no available relative times for dataset-specific time sampling."
            raise ValueError(msg)

        candidate_times = [value for value in available_times if value <= int(relative_time)]
        if not candidate_times:
            msg = (
                f"Dataset '{dataset_name}' has no forcing/boundary time at or before relative time "
                f"{relative_time}. Available times: {available_times}"
            )
            raise ValueError(msg)
        sampled_time = candidate_times[-1]

        LOGGER.info(
            "Dataset time sampling dataset=%s requested_time=%s sampled_time=%s",
            dataset_name,
            relative_time,
            sampled_time,
        )

        return int(time_map[sampled_time])

    def get_inputs(
        self,
        batch: dict[str, torch.Tensor],
        data_indices: dict[str, IndexCollection],
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        """Extract model inputs from a batch.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Full batch keyed by dataset name.
        data_indices : dict[str, IndexCollection]
            Data indices per dataset.

        Returns
        -------
        dict[str, torch.Tensor]
            Input tensors per dataset with shape
            ``(bs, num_inputs, grid, nvar)``.
        """
        if len(self.dataset_time_maps) == 0:
            x = {}
            for dataset_name, dataset_batch in batch.items():
                time_indices = self.get_batch_input_indices(dataset_name=dataset_name)
                time_indices = normalize_time_indices(time_indices)
                dataset_batch = dataset_batch[:, time_indices]
                x[dataset_name] = dataset_batch[..., data_indices[dataset_name].data.input.full]
                LOGGER.debug("SHAPE: x[%s].shape = %s", dataset_name, list(x[dataset_name].shape))
            return x

        x = {}
        for dataset_name, dataset_batch in batch.items():
            requested_relative_times = self._requested_input_relative_times(dataset_name)
            input_positions = [
                self._sample_batch_position(dataset_name=dataset_name, relative_time=relative_time)
                for relative_time in requested_relative_times
            ]
            input_index = torch.tensor(input_positions, device=dataset_batch.device, dtype=torch.long)
            x_time = dataset_batch.index_select(1, input_index)
            x[dataset_name] = x_time[..., data_indices[dataset_name].data.input.full]
            LOGGER.debug("SHAPE: x[%s].shape = %s", dataset_name, list(x[dataset_name].shape))
        return x

    def get_targets(self, batch: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """Extract model targets from a batch.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Full batch keyed by dataset name.
        **kwargs
            Forwarded to ``get_batch_output_indices`` (e.g.
            ``rollout_step``).

        Returns
        -------
        dict[str, torch.Tensor]
            Target tensors per dataset with shape
            ``(bs, num_outputs, ensemble, grid, full_nvar)`` in DATA_FULL
            variable space (all variables including forcings).
        """
        if len(self.dataset_time_maps) == 0:
            y = {}
            for dataset_name, dataset_batch in batch.items():
                time_indices = self.get_batch_output_indices(dataset_name=dataset_name, **kwargs)
                time_indices = normalize_time_indices(time_indices)
                y[dataset_name] = dataset_batch[:, time_indices]
                LOGGER.debug("SHAPE: y[%s].shape = %s", dataset_name, list(y[dataset_name].shape))
            return y

        y = {}
        for dataset_name, dataset_batch in batch.items():
            requested_relative_times = self._requested_output_relative_times(dataset_name, **kwargs)
            target_positions = [
                self._sample_batch_position(dataset_name=dataset_name, relative_time=relative_time)
                for relative_time in requested_relative_times
            ]
            target_index = torch.tensor(target_positions, device=dataset_batch.device, dtype=torch.long)
            y[dataset_name] = dataset_batch.index_select(1, target_index)
            LOGGER.debug("SHAPE: y[%s].shape = %s", dataset_name, list(y[dataset_name].shape))
        return y

    def log_extra(self, *_args, **_kwargs) -> None:  # noqa: B027
        """Hook to log any task-specific information."""

    def on_train_epoch_end(self, current_epoch: int) -> None:  # noqa: B027
        """Hook to update task state at the end of each training epoch (e.g. for curriculum learning)."""

    def fill_metadata(self, md_dict: dict) -> None:
        """Fill the metadata dictionary with task-specific information."""
        md_dict["task"] = self.name

        metadata_inference = md_dict.get("metadata_inference", {})
        dataset_names = metadata_inference.get("dataset_names", []) if isinstance(metadata_inference, Mapping) else []

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

        for dataset_name in dataset_names:
            existing_timesteps = metadata_inference[dataset_name].get("timesteps", {})
            metadata_inference[dataset_name]["timesteps"] = timesteps | existing_timesteps

    def configure_from_metadata(self, md_dict: dict) -> None:
        """Initialize task runtime state from metadata."""
        metadata_inference = md_dict.get("metadata_inference", {})
        dataset_names = metadata_inference.get("dataset_names", []) if isinstance(metadata_inference, Mapping) else []

        self.dataset_input_relative_times_by_dataset = {}
        self.dataset_target_relative_times_by_dataset = {}
        self.dataset_time_maps = {}

        if len(dataset_names) == 0:
            return

        self.dataset_input_relative_times_by_dataset = self._resolve_relative_time_metadata(
            metadata_inference,
            dataset_names,
            (
                "relative_date_input_indices_validation_by_dataset",
                "relative_date_input_indices_training_by_dataset",
            ),
        )
        self.dataset_target_relative_times_by_dataset = self._resolve_relative_time_metadata(
            metadata_inference,
            dataset_names,
            (
                "relative_date_target_indices_validation_by_dataset",
                "relative_date_target_indices_training_by_dataset",
            ),
        )
        relative_by_dataset = self._resolve_relative_time_metadata(
            metadata_inference,
            dataset_names,
            (
                "relative_date_indices_validation_by_dataset",
                "relative_date_indices_training_by_dataset",
            ),
        )
        dataset_time_maps = {}
        for dataset_name in dataset_names:
            relative_times = relative_by_dataset.get(dataset_name)
            if relative_times is None:
                relative_times = sorted(
                    {
                        *self._requested_input_relative_times(dataset_name),
                        *self._requested_output_relative_times(
                            dataset_name,
                            step=max(self.num_steps - 1, 0),
                        ),
                    },
                )

            dataset_time_maps[dataset_name] = {
                int(relative_time): batch_idx for batch_idx, relative_time in enumerate(relative_times)
            }

        self.dataset_time_maps = dataset_time_maps


class BaseSingleStepTask(BaseTask):
    """Base class for single-step tasks."""

    def advance_input(self, *args, **_kwargs) -> dict[str, torch.Tensor]:
        """Advance the input state for each dataset based on the task's requirements.

        This method can be overridden by specific tasks to implement custom logic for advancing the input state.

        Returns
        -------
            dict[str, torch.Tensor]: The advanced input state for each dataset.
        """
        return args[0]
