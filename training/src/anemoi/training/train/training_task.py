# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from abc import ABC, abstractmethod
from anemoi.utils.dates import frequency_to_seconds
import datetime
import torch
from anemoi.models.data_indices.collection import IndexCollection

LOGGER = logging.getLogger(__name__)


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
    def get_inputs(
        self, batch: dict[str, torch.Tensor], data_indices: dict[str, IndexCollection]
    ) -> dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def get_targets(
        self, batch: dict[str, torch.Tensor], data_indices: dict[str, IndexCollection]
    ) -> dict[str, torch.Tensor]:
        pass


class ForecastingTask(BaseTask):
    """Forecasting task implementation."""

    def __init__(self, multistep_input: int, multistep_output: int, timestep: str, **kwargs) -> None:
        super().__init__(timestep)
        self.num_input_steps = multistep_input
        self.num_output_steps = multistep_output

    def get_relative_input_time_indices(self, frequency: str | datetime.timedelta) -> list[int]:
        """Get the relative time indices for the model input sequence.

        Returns:
            list[int]: List of relative time indices.
        """
        return list(range(0, self.timeincrement(frequency) * self.num_input_steps, self.timeincrement(frequency)))

    def get_relative_target_time_indices(self, frequency: str | datetime.timedelta) -> list[int]:
        """Get the relative time indices for the model target sequence.

        Returns:
            list[int]: List of relative time indices.
        """
        start = self.timeincrement(frequency) * self.num_input_steps
        return list(range(
            start,
            start + self.timeincrement(frequency) * self.num_output_steps,
            self.timeincrement(frequency),
        ))

    def get_relative_time_indices(self, frequency: str | datetime.timedelta) -> list[int]:
        """Get the relative time indices for the model input sequence.

        Returns:
            list[int]: List of relative time indices.
        """
        return self.get_relative_input_time_indices(frequency) + self.get_relative_target_time_indices(frequency)

    def get_inputs(self, batch: dict[str, torch.Tensor], data_indices: dict[str, IndexCollection]) -> dict[str, torch.Tensor]:
        timesteps = slice(0, self.num_input_steps)
        x = {}
        for dataset_name, dataset_batch in batch.items():
            x[dataset_name] = dataset_batch[:, timesteps, ..., data_indices[dataset_name].data.input.full]
            # (bs, multi_step, latlon, nvar)
            LOGGER.debug("SHAPE: x[%s].shape = %s", dataset_name, list(x[dataset_name].shape))
        return x

    def get_targets(self, batch: dict[str, torch.Tensor], data_indices: dict[str, IndexCollection], step) -> dict[str, torch.Tensor]:
        timesteps = self.num_input_steps # slice(self.num_input_steps, self.num_input_steps + self.num_output_steps)
        y = {}
        for dataset_name, dataset_batch in batch.items():
            y[dataset_name] = dataset_batch[:, timesteps, ..., data_indices[dataset_name].data.output.full]
            LOGGER.debug("SHAPE: y[%s].shape = %s", dataset_name, list(y[dataset_name].shape))
        return y

    def _advance_dataset_input(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        batch: torch.Tensor,
        rollout_step: int = 0,
        data_indices: IndexCollection | None = None,
    ) -> torch.Tensor:
        x = x.roll(-1, dims=1)

        # Get prognostic variables
        x[:, -1, :, :, data_indices.model.input.prognostic] = y_pred[
            ...,
            data_indices.model.output.prognostic,
        ]

        #x[:, -1] = self.output_mask[dataset_name].rollout_boundary(
        #    x[:, -1],
        #    batch[:, self.multi_step + rollout_step],
        #    data_indices[dataset_name],
        #    grid_shard_slice=self.grid_shard_slice[dataset_name],
        #)

        # get new "constants" needed for time-varying fields
        x[:, -1, :, :, data_indices.model.input.forcing] = batch[
            :,
            self.multi_step + rollout_step,
            :,
            :,
            data_indices.data.input.forcing,
        ]
        return x

    def advance_input(self, x, y_pred, batch, rollout_step=None, data_indices=None):
        """Advance the input state for the next rollout step."""
        for dataset_name in x:
            x[dataset_name] = self._advance_dataset_input(
                x[dataset_name],
                y_pred[dataset_name],
                batch[dataset_name],
                rollout_step=rollout_step,
                data_indices=data_indices[dataset_name],
            )
        return x

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




