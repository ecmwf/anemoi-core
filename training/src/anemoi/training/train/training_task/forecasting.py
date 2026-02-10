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

    def __init__(self, multistep_input: int, multistep_output: int, timestep: str, **_kwargs) -> None:
        super().__init__(timestep)
        self.num_input_steps = multistep_input
        self.num_output_steps = multistep_output

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
        x = x.roll(-1, dims=1)

        # Get prognostic variables
        x[:, -1, :, :, data_indices.model.input.prognostic] = y_pred[
            ...,
            data_indices.model.output.prognostic,
        ]

        # x[:, -1] = self.output_mask[dataset_name].rollout_boundary(
        #    x[:, -1],
        #    batch[:, self.multi_step + rollout_step],
        #    data_indices[dataset_name],
        #    grid_shard_slice=self.grid_shard_slice[dataset_name],
        # )

        # get new "constants" needed for time-varying fields
        x[:, -1, :, :, data_indices.model.input.forcing] = batch[
            :,
            self.multi_step + rollout_step,
            :,
            :,
            data_indices.data.input.forcing,
        ]
        return x

    def advance_input(
        self,
        x: dict[str, torch.Tensor],
        y_pred: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        rollout_step: int | None = None,
        data_indices: IndexCollection | None = None,
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
