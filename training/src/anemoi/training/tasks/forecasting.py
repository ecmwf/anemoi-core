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
from functools import cached_property

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.tasks.base import BaseTask
from anemoi.utils.dates import frequency_to_seconds

LOGGER = logging.getLogger(__name__)


class ForecastingTask(BaseTask):
    """Forecasting task implementation."""

    name: str = "forecasting"

    def __init__(
        self,
        multistep_input: int,
        multistep_output: int,
        timestep: str,
        rollout_start: int = 1,
        rollout_epoch_increment: int = 0,
        rollout_max: int = 1,
        **_kwargs,
    ) -> None:
        self.timestep = frequency_to_seconds(timestep)

        self.num_input_steps = multistep_input
        self.num_output_steps = multistep_output

        self.rollout = rollout_start
        self.rollout_epoch_increment = rollout_epoch_increment
        self.rollout_max = rollout_max

    def timeincrement(self, frequency: str | datetime.timedelta) -> int:
        freq_seconds = frequency_to_seconds(frequency)
        return self.timestep // freq_seconds

    def get_batch_input_time_indices(self, *args, **kwargs) -> list[int]:
        return list(range(self.num_input_steps))

    def get_batch_output_time_indices(self, *args, **kwargs) -> list[int]:
        return list(range(self.num_input_steps, self.num_input_steps + self.num_output_steps))

    def get_dataset_input_time_indices(self, frequency: str | datetime.timedelta) -> list[int]:
        """Get the relative time indices for the model input sequence.

        Returns
        -------
            list[int]: List of relative time indices.
        """
        return list(range(0, self.timeincrement(frequency) * self.num_input_steps, self.timeincrement(frequency)))

    def get_dataset_target_time_indices(self, frequency: str | datetime.timedelta) -> list[int]:
        """Get the relative time indices for the model target sequence.

        Returns
        -------
            list[int]: List of relative time indices.
        """
        start = self.timeincrement(frequency) * self.num_input_steps
        return list(
            range(
                start,
                start + self.timeincrement(frequency) * self.num_output_steps,
                self.timeincrement(frequency),
            ),
        )

    def get_relative_time_indices(self, frequency: str | datetime.timedelta) -> list[int]:
        """Get the relative time indices for the model input sequence.

        Returns
        -------
            list[int]: List of relative time indices.
        """
        return self.get_dataset_input_time_indices(frequency) + self.get_dataset_target_time_indices(frequency)

    def _advance_dataset_input(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        batch: torch.Tensor,
        rollout_step: int = 0,
        data_indices: IndexCollection | None = None,
    ) -> torch.Tensor:
        """Default implementation used by simple rollout tasks.

        Supports model outputs shaped like:
        - (B, T, E, G, V)
        """
        keep_steps = min(self.num_input_steps, self.num_output_steps)

        x = x.roll(-keep_steps, dims=1)

        # TODO(dieter): see if we can replace for loop with tensor operations
        for i in range(keep_steps):
            # Get prognostic variables
            x[:, -(i + 1), ..., data_indices.model.input.prognostic] = y_pred[
                :,
                -(i + 1),
                ...,
                data_indices.model.output.prognostic,
            ]

            batch_time_index = self.num_input_steps + (rollout_step + 1) * self.num_output_steps - (i + 1)

            # x[:, -(i + 1)] = self.output_mask[dataset_name].rollout_boundary(
            #    x[:, -(i + 1)],
            #    batch[:, batch_time_index],
            #    data_indices,
            #    grid_shard_slice=self.grid_shard_slice[dataset_name],
            # )

            # get new "constants" needed for time-varying fields
            x[:, -(i + 1), ..., data_indices.model.input.forcing] = batch[
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
    ) -> dict[str, torch.Tensor]:
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

    @cached_property
    def steps(self) -> tuple[dict[str, int]]:
        """Get the range of rollout steps to perform."""
        return tuple({"rollout_step": i} for i in range(self.rollout))

    def log_extra(self, logger, logger_enabled) -> None:
        """Log any task-specific information."""
        logger(
            "rollout",
            float(self.rollout),
            on_step=True,
            logger=logger_enabled,
            rank_zero_only=True,
            sync_dist=False,
        )

    def on_train_epoch_end(self, current_epoch: int) -> None:
        if self.rollout_epoch_increment > 0 and current_epoch % self.rollout_epoch_increment == 0:
            self.rollout += 1
            LOGGER.debug("Rollout window length: %d", self.rollout)
        self.rollout = min(self.rollout, self.rollout_max)
