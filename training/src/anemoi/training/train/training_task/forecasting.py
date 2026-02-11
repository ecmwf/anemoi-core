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

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.train.training_task.base import BaseTask

LOGGER = logging.get_logger(__name__)


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
        super().__init__(timestep)
        self.num_input_steps = multistep_input
        self.num_output_steps = multistep_output

        self.rollout = rollout_start
        self.rollout_epoch_increment = rollout_epoch_increment
        self.rollout_max = rollout_max

    def get_relative_input_time_indices(self, frequency: str | datetime.timedelta) -> list[int]:
        """Get the relative time indices for the model input sequence.

        Returns
        -------
            list[int]: List of relative time indices.
        """
        return list(range(0, self.timeincrement(frequency) * self.num_input_steps, self.timeincrement(frequency)))

    def get_relative_target_time_indices(self, frequency: str | datetime.timedelta) -> list[int]:
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
        return self.get_relative_input_time_indices(frequency) + self.get_relative_target_time_indices(frequency)

    def get_inputs(
        self,
        batch: dict[str, torch.Tensor],
        data_indices: dict[str, IndexCollection],
    ) -> dict[str, torch.Tensor]:
        timesteps = slice(0, self.num_input_steps)
        x = {}
        for dataset_name, dataset_batch in batch.items():
            x[dataset_name] = dataset_batch[:, timesteps, ..., data_indices[dataset_name].data.input.full]
            # shape: (bs, multi_step, latlon, nvar)
            LOGGER.debug("SHAPE: x[%s].shape = %s", dataset_name, list(x[dataset_name].shape))
        return x

    def get_targets(
        self,
        batch: dict[str, torch.Tensor],
        data_indices: dict[str, IndexCollection],
        step: int,
    ) -> dict[str, torch.Tensor]:
        start = self.num_input_steps + self.num_output_steps * step
        y = {}
        for dataset_name, dataset_batch in batch.items():
            y_time = dataset_batch.narrow(1, start, self.num_output_steps)
            var_indices = data_indices[dataset_name].data.output.full.to(device=dataset_batch.device)
            y[dataset_name] = y_time.index_select(-1, var_indices)
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

            #x[:, -(i + 1)] = self.output_mask[dataset_name].rollout_boundary(
            #    x[:, -(i + 1)],
            #    batch[:, batch_time_index],
            #    data_indices,
            #    grid_shard_slice=self.grid_shard_slice[dataset_name],
            #)

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
        step: int | None = None,
        data_indices: IndexCollection | None = None,
    ) -> dict[str, torch.Tensor]:
        """Advance the input state for the next rollout step."""
        for dataset_name in x:
            x[dataset_name] = self._advance_dataset_input(
                x[dataset_name],
                y_pred[dataset_name],
                batch[dataset_name],
                rollout_step=step,
                data_indices=data_indices[dataset_name],
            )
        return x

    def steps(self) -> range:
        """Get the range of rollout steps to perform."""
        return range(self.rollout)

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
