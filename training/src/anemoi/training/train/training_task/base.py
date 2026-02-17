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
from abc import abstractmethod
from collections.abc import Iterable
from functools import cached_property

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.utils.dates import frequency_to_seconds

LOGGER = logging.getLogger(__name__)


class BaseTask(ABC):
    """Base class for all tasks.

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

    def __init__(self, timestep: str, **_kwargs) -> None:
        self.timestep = frequency_to_seconds(timestep)

    def timeincrement(self, frequency: str | datetime.timedelta) -> int:
        freq_seconds = frequency_to_seconds(frequency)
        return self.timestep // freq_seconds

    @abstractmethod
    def get_batch_input_time_indices(self, *args, **kwargs) -> list[int]:
        """Get the relative time indices for the model input sequence.

        Returns
        -------
            list[int]: List of relative time indices.
        """
        raise NotImplementedError

    @abstractmethod
    def get_batch_output_time_indices(self, *args, **kwargs) -> list[int]:
        """Get the relative time indices for the model target sequence.

        By default, this is the same as the input time indices, but it can be overridden by specific tasks.

        Returns
        -------
            list[int]: List of relative time indices.
        """
        raise NotImplementedError

    @abstractmethod
    def get_relative_time_indices(self, *args, **kwargs) -> list[int]:
        """Get the relative time indices for the model input sequence.

        Returns
        -------
            list[int]: List of relative time indices.
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def steps(self) -> Iterable[dict]:
        """Get the steps for the task."""
        raise NotImplementedError

    @property
    def num_inputs(self) -> int:
        """Get the number of input time steps for the task."""
        return len(self.get_batch_input_time_indices())

    @property
    def num_outputs(self) -> int:
        """Get the number of output time steps for the task."""
        return len(self.get_batch_output_time_indices())

    @property
    def num_steps(self) -> int:
        """Get the number of steps in the task."""
        return len(self.steps)

    def get_inputs(
        self,
        batch: dict[str, torch.Tensor],
        data_indices: dict[str, IndexCollection],
    ) -> dict[str, torch.Tensor]:
        time_indices = self.get_batch_input_time_indices()

        x = {}
        for dataset_name, dataset_batch in batch.items():
            dataset_batch = dataset_batch[:, time_indices]
            x[dataset_name] = dataset_batch[..., data_indices[dataset_name].data.input.full]
            # shape: (bs, multi_step, latlon, nvar)
            LOGGER.debug("SHAPE: x[%s].shape = %s", dataset_name, list(x[dataset_name].shape))
        return x

    def get_targets(
        self,
        batch: dict[str, torch.Tensor],
        data_indices: dict[str, IndexCollection],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        time_indices = self.get_batch_output_time_indices(**kwargs)

        y = {}
        for dataset_name, dataset_batch in batch.items():
            dataset_batch = dataset_batch[:, time_indices]
            var_indices = data_indices[dataset_name].data.output.full.to(device=dataset_batch.device)
            y[dataset_name] = dataset_batch[..., var_indices]
            LOGGER.debug("SHAPE: y[%s].shape = %s", dataset_name, list(y[dataset_name].shape))
        return y


    def fill_metadata(self, md_dict: dict) -> None:
        """Fill the metadata dictionary with task-specific information.

        Args:
            md_dict (dict): The metadata dictionary to fill.
        """
        md_dict["task"] = self.name
        # md_dict["timestep"] = self.timestep
        # Save relative time indices
        md_dict["relative_input_time_indices"] = self.get_batch_input_time_indices()
        md_dict["relative_output_time_indices"] = self.get_batch_output_time_indices()


class BaseTimelessTask(BaseTask):
    """Base class for timeless tasks."""

    def __init__(self, **_kwargs) -> None:
        pass

    def get_relative_time_indices(self, *_args, **_kwargs) -> list[int]:
        """Get the relative time indices for the model input sequence.

        Returns
        -------
            list[int]: List of relative time indices.
        """
        return [0]

    def get_batch_input_time_indices(self, *args, **kwargs) -> list[int]:
        return [0]

    def get_batch_output_time_indices(self, *args, **kwargs) -> list[int]:
        return [0]

    @cached_property
    def steps(self) -> Iterable[dict]:
        """Get the steps for the task."""
        return ({},)
