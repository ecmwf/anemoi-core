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
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

LOGGER = logging.getLogger(__name__)


class BaseTask(ABC):
    """Base class for all tasks.

    Tasks define the temporal structure of a training sample by specifying
    input and output time offsets as ``timedelta`` objects. The base class
    provides:

    * An ``offset`` property that is the union of input and output offsets,
      used by the datamodule to determine which time steps to load.
    * ``get_inputs`` / ``get_targets`` to split a loaded batch into model
      inputs and targets based on position within ``offset``.
    * ``get_relative_time_indices(frequency)`` to convert offsets into
      integer indices for a specific dataset frequency.

    Attributes
    ----------
    name : str
        Name of the task.
    num_inputs : int
        Number of input time steps for the task.
    num_outputs : int
        Number of output time steps for the task.
    num_steps : int
        Number of steps in the task (for multi-step tasks).
    """

    name: str

    def __init__(
        self,
        inputs_offsets: list[datetime.timedelta],
        outputs_offsets: list[datetime.timedelta],
        steps: Iterable[dict] | None = None,
    ) -> None:
        self._inputs_offsets = sorted(inputs_offsets)
        self._outputs_offsets = sorted(outputs_offsets)
        self._steps = steps if steps is not None else ({},)

    @property
    def inputs_offsets(self) -> list[datetime.timedelta]:
        """Sorted input time offsets."""
        return self._inputs_offsets

    @property
    def outputs_offsets(self) -> list[datetime.timedelta]:
        """Sorted output time offsets for the current step.

        Subclasses may override this to support parametrised output
        selection (e.g. per rollout step).
        """
        return self._outputs_offsets

    @property
    def steps(self) -> Iterable[dict]:
        """Get the steps for the task."""
        return self._steps

    @property
    def offset(self) -> list[datetime.timedelta]:
        """Full sorted union of input and output offsets.

        This is used by the datamodule to compute
        ``data_relative_time_indices``.
        """
        return sorted(set(self._inputs_offsets + self._outputs_offsets))

    @property
    def num_input_timesteps(self) -> int:
        """Number of input time steps."""
        return len(self._inputs_offsets)

    @property
    def num_output_timesteps(self) -> int:
        """Number of output time steps."""
        return len(self._outputs_offsets)

    @property
    def num_steps(self) -> int:
        """Number of steps in the task."""
        return len(self._steps)

    def _offset_to_batch_indices(self, offsets: list[datetime.timedelta]) -> list[int]:
        """Map a list of offsets to their positions in ``self.offset``."""
        full = self.offset
        return [full.index(o) for o in offsets]

    def get_relative_time_indices(self, frequency: str | datetime.timedelta) -> list[int]:
        """Convert ``self.offset`` to integer dataset indices for a given *frequency*.

        The smallest offset is mapped to index 0.

        Parameters
        ----------
        frequency : str | datetime.timedelta
            Data frequency of the dataset (e.g. ``"1H"`` or ``timedelta(hours=1)``).

        Returns
        -------
        list[int]
            Integer indices suitable for indexing into the dataset's time
            dimension.
        """
        if not isinstance(frequency, datetime.timedelta):
            frequency = frequency_to_timedelta(frequency)

        return [o // frequency for o in self.offset]

    def get_batch_input_indices(self) -> list[int]:
        """Positions of the input offsets within the full batch ``offset``."""
        return self._offset_to_batch_indices(self._inputs_offsets)

    def get_batch_output_indices(self, **kwargs) -> list[int]:
        """Positions of the output offsets within the full batch ``offset``.

        Parameters are forwarded to ``get_output_offset`` so that
        subclasses can parametrise the output selection (e.g. per
        rollout step).
        """
        return self._offset_to_batch_indices(self.get_output_offsets(**kwargs))

    def get_output_offsets(self, **_kwargs) -> list[datetime.timedelta]:
        """Return the output offsets for a given step.

        The default implementation returns ``self.outputs_offsets``.
        Subclasses may override this to shift outputs per rollout step.
        """
        return self.outputs_offsets

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
        time_indices = self.get_batch_input_indices()

        x = {}
        for dataset_name, dataset_batch in batch.items():
            dataset_batch = dataset_batch[:, time_indices]
            x[dataset_name] = dataset_batch[..., data_indices[dataset_name].data.input.full]
            LOGGER.debug("SHAPE: x[%s].shape = %s", dataset_name, list(x[dataset_name].shape))
        return x

    def get_targets(
        self,
        batch: dict[str, torch.Tensor],
        data_indices: dict[str, IndexCollection],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Extract model targets from a batch.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Full batch keyed by dataset name.
        data_indices : dict[str, IndexCollection]
            Data indices per dataset.
        **kwargs
            Forwarded to ``get_batch_output_indices`` (e.g.
            ``rollout_step``).

        Returns
        -------
        dict[str, torch.Tensor]
            Target tensors per dataset with shape
            ``(bs, num_outputs, grid, nvar)``.
        """
        time_indices = self.get_batch_output_indices(**kwargs)

        y = {}
        for dataset_name, dataset_batch in batch.items():
            dataset_batch = dataset_batch[:, time_indices]
            var_indices = data_indices[dataset_name].data.output.full.to(device=dataset_batch.device)
            y[dataset_name] = dataset_batch[..., var_indices]
            LOGGER.debug("SHAPE: y[%s].shape = %s", dataset_name, list(y[dataset_name].shape))
        return y

    def log_extra(self, *_args, **_kwargs) -> None:
        """Hook to log any task-specific information."""

    def on_train_epoch_end(self, current_epoch: int) -> None:
        """Hook to update task state at the end of each training epoch (e.g. for curriculum learning)."""

    def fill_metadata(self, md_dict: dict) -> None:
        """Fill the metadata dictionary with task-specific information."""
        md_dict["task"] = self.name
        md_dict["input_offsets"] = [frequency_to_string(o) for o in self.inputs_offsets]
        md_dict["output_offsets"] = [frequency_to_string(o) for o in self.outputs_offsets]
        md_dict["num_input_timesteps"] = self.num_input_timesteps
        md_dict["num_output_timesteps"] = self.num_output_timesteps


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
