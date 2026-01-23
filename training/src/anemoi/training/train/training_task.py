# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import ABC, abstractmethod
from anemoi.utils.dates import frequency_to_seconds
import datetime


class BaseTask(ABC):
    """Base class for all tasks."""

    def __init__(self, timestep: str, **_kwargs) -> None:
        self.timestep = frequency_to_seconds(timestep)

    def timeincrement(self, frequency: str | datetime.timedelta) -> int:
        freq_seconds = frequency_to_seconds(frequency)
        return self.timestep // freq_seconds

    @abstractmethod
    def get_relative_time_indices(self, *args, **kwargs) -> list[int]:
        """Get the relative time indices for the model input sequence.

        Returns:
            list[int]: List of relative time indices.
        """
        raise NotImplementedError

    @abstractmethod
    def get_inputs(self, index: int):
        pass

    @abstractmethod
    def get_targets(self, index: int):
        pass


class ForecastingTask(BaseTask):
    """Forecasting task implementation."""

    def __init__(self, multistep_input: int, multistep_output: int, timestep: str, rollout: dict | None = None) -> None:
        super().__init__(timestep)
        self.num_input_steps = multistep_input
        self.num_output_steps = multistep_output
        self.rollout = rollout

    def get_relative_input_time_indices(self, frequency: str | datetime.timedelta) -> list[int]:
        """Get the relative time indices for the model input sequence.

        Returns:
            list[int]: List of relative time indices.
        """
        return list(range(self.timeincrement(frequency) * self.num_input_steps))

    def get_relative_target_time_indices(self, frequency: str | datetime.timedelta) -> list[int]:
        """Get the relative time indices for the model target sequence.

        Returns:
            list[int]: List of relative time indices.
        """
        start = self.timeincrement(frequency) * self.num_input_steps
        return list(range(start, start + self.timeincrement(frequency) * self.num_output_steps))

    def get_relative_time_indices(self, frequency: str | datetime.timedelta) -> list[int]:
        """Get the relative time indices for the model input sequence.

        Returns:
            list[int]: List of relative time indices.
        """
        return self.get_relative_input_time_indices(frequency) + self.get_relative_target_time_indices(frequency)

    def get_batch_time_indices(self, index: int, frequency: str | datetime.timedelta) -> list[int]:
        """Get the time indices for the model input and target sequences.

        Returns:
            list[int]: List of time indices.
        """
        return [index + i for i in self.get_relative_time_indices(frequency)]

    def get_inputs(self, index: int, frequency: str | datetime.timedelta):
        return [index + i for i in self.get_relative_input_time_indices(frequency)]

    def get_targets(self, index: int, frequency: str | datetime.timedelta):
        return [index + i for i in self.get_relative_target_time_indices(frequency)]


class BaseExplicitIndicesTask(BaseTask):
    """Base class for tasks with explicit time indices."""

    def __init__(self, input_times: list[int], target_times: list[int], **_kwargs):
        self.input_times = input_times
        self.target_times = target_times

    def get_relative_time_indices(self) -> list[int]:
        """Get the relative time indices for the model input sequence.

        Returns:
            list[int]: List of relative time indices.
        """
        return sorted(set(self.input_times + self.target_times))

    def get_inputs(self, index: int):
        return [index + t for t in self.input_times]

    def get_targets(self, index: int):
        return [index + t for t in self.target_times]


class TimeInterpolationTask(BaseExplicitIndicesTask):
    """Time interpolation task implementation."""


class BaseTimelessTask(BaseTask):
    """Base class for timeless tasks."""

    def get_relative_time_indices(self) -> list[int]:
        """Get the relative time indices for the model input sequence.

        Returns:
            list[int]: List of relative time indices.
        """
        return [0]

    def get_inputs(self, index: int):
        return index

    def get_targets(self, index: int):
        return index


class AutoencodingTask(BaseTimelessTask):
    """Autoencoding task implementation."""


class DownscalingTask(BaseTimelessTask):
    """Downscaling task implementation."""




