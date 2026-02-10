# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from abc import ABC
from abc import abstractmethod

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.utils.dates import frequency_to_seconds


class BaseTask(ABC):
    """Base class for all tasks."""

    name: str

    def __init__(self, timestep: str, **_kwargs) -> None:
        self.timestep = frequency_to_seconds(timestep)

    def timeincrement(self, frequency: str | datetime.timedelta) -> int:
        freq_seconds = frequency_to_seconds(frequency)
        return self.timestep // freq_seconds

    @abstractmethod
    def get_relative_time_indices(self, *args, **kwargs) -> list[int]:
        """Get the relative time indices for the model input sequence.

        Returns
        -------
            list[int]: List of relative time indices.
        """
        raise NotImplementedError

    @abstractmethod
    def get_inputs(
        self,
        batch: dict[str, torch.Tensor],
        data_indices: dict[str, IndexCollection],
    ) -> dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def get_targets(
        self,
        batch: dict[str, torch.Tensor],
        data_indices: dict[str, IndexCollection],
    ) -> dict[str, torch.Tensor]:
        pass

    def fill_metadata(self, md_dict: dict) -> None:
        """Fill the metadata dictionary with task-specific information.

        Args:
            md_dict (dict): The metadata dictionary to fill.
        """
        md_dict["task"] = self.name
        md_dict["timestep"] = self.timestep
        # Save relative time indices


class BaseTimelessTask(BaseTask):
    """Base class for timeless tasks."""

    def get_relative_time_indices(self) -> list[int]:
        """Get the relative time indices for the model input sequence.

        Returns
        -------
            list[int]: List of relative time indices.
        """
        return [0]

    def get_inputs(self, index: int) -> list[int]:
        return index

    def get_targets(self, index: int) -> list[int]:
        return index
