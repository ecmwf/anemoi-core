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
        input_offsets: list[datetime.timedelta],
        output_offsets: list[datetime.timedelta],
    ) -> None:
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
        time_indices = self.get_batch_input_indices()
        time_indices = normalize_time_indices(time_indices)

        var_indices = {
            dataset_name: data_indices[dataset_name].data.input.full for dataset_name in batch.dataset_names
        }
        new_batch = batch.select_time(time_indices).select_vars(var_indices)
        for dataset_name, payload in new_batch.data.items():
            LOGGER.debug(
                "SHAPE: x[%s] = %s",
                dataset_name,
                payload.shape if hasattr(payload, "shape") else [t.shape for t in payload],
            )
        return new_batch

    def get_targets(self, batch: "Batch", **kwargs) -> "Batch":
        """Extract model targets from a Batch, preserving coords and metadata.

        Parameters
        ----------
        batch : Batch
            Full batch object (data, coords, metadata) keyed by dataset name.
        **kwargs
            Forwarded to ``get_batch_output_indices`` (e.g. ``rollout_step``).

        Returns
        -------
        Batch
            New Batch with target tensors per dataset (shape ``(bs, num_outputs, ensemble, grid, full_nvar)``),
            sharing coords and metadata by reference.
        """
        time_indices = self.get_batch_output_indices(**kwargs)
        time_indices = normalize_time_indices(time_indices)

        new_batch = batch.select_time(time_indices)
        for dataset_name, payload in new_batch.data.items():
            LOGGER.debug(
                "SHAPE: y[%s] = %s",
                dataset_name,
                payload.shape if hasattr(payload, "shape") else [t.shape for t in payload],
            )
        return new_batch

    def log_extra(self, *_args, **_kwargs) -> None:  # noqa: B027
        """Hook to log any task-specific information."""

    def on_train_epoch_end(self, current_epoch: int) -> None:  # noqa: B027
        """Hook to update task state at the end of each training epoch (e.g. for curriculum learning)."""

    def fill_metadata(self, md_dict: dict) -> None:
        """Fill the metadata dictionary with task-specific information."""
        md_dict["task"] = self.name

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
            md_dict["metadata_inference"][dataset_name]["timesteps"] = timesteps


class BaseSingleStepTask(BaseTask):
    """Base class for single-step tasks."""

    def advance_input(self, batch: "Batch", *args, **_kwargs) -> "Batch":
        """Advance the input state for each dataset based on the task's requirements.

        This method can be overridden by specific tasks to implement custom logic for advancing the input state.

        Returns
        -------
            Batch: The advanced input state for each dataset (default: passthrough).
        """
        return batch
