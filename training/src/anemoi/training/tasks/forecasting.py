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
from anemoi.training.tasks.base import BaseTask
from anemoi.utils.dates import frequency_to_string, frequency_to_timedelta

LOGGER = logging.getLogger(__name__)


class ForecastingTask(BaseTask):
    """Forecasting task implementation.

    Builds input and output offsets from ``multistep_input``,
    ``multistep_output`` and a ``timestep`` string (e.g. ``"6H"``).

    For rollout training the ``offset`` property extends the output
    offsets up to ``rollout_max`` steps so the datamodule loads enough
    time steps, while ``steps`` only iterates over the current
    ``rollout`` value which grows via ``on_train_epoch_end``.
    """

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
        self.timestep = frequency_to_timedelta(timestep)

        self.num_input_steps = multistep_input
        self.num_output_steps = multistep_output

        self.rollout = rollout_start
        self.rollout_epoch_increment = rollout_epoch_increment
        self.rollout_max = rollout_max

        # Build offsets from multistep configuration
        # Input: e.g. multistep_input=2, timestep=6H    ->  [-6H, 0H]
        inputs_offsets = [-1 * i * self.timestep for i in range(multistep_input)]
        # Outputs: e.g. multistep_output=1, timestep=6H  -> [[6H], [12H], [18H], ...] up to rollout_max
        outputs_offsets = [(i + 1) * self.timestep for i in range(multistep_output)]
        super().__init__(inputs_offsets=inputs_offsets, outputs_offsets=outputs_offsets)

    # ------------------------------------------------------------------
    # Offset overrides for rollout
    # ------------------------------------------------------------------

    @property
    def _step_shift(self) -> datetime.timedelta:
        """Time shift between consecutive rollout steps."""
        return self.timestep * self.num_output_steps

    @property
    def offset(self) -> list[datetime.timedelta]:
        """Union of input + all rollout output offsets up to ``rollout_max``.

        Pre-allocates enough batch timesteps so the datamodule always
        loads data for the maximum rollout length.
        """
        all_offsets = set(self._inputs_offsets)
        for step in range(self.rollout_max):
            shift = self._step_shift * step
            for o in self._outputs_offsets:
                all_offsets.add(o + shift)
        return sorted(all_offsets)

    def get_output_offset(self, rollout_step: int = 0, **_kwargs) -> list[datetime.timedelta]:
        """Return output offsets shifted by ``rollout_step``."""
        shift = self._step_shift * rollout_step
        return sorted(o + shift for o in self._outputs_offsets)

    @property
    def steps(self) -> tuple[dict[str, int], ...]:
        """Get the range of rollout steps to perform."""
        return tuple({"rollout_step": i} for i in range(self.rollout))

    def _advance_dataset_input(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        batch: torch.Tensor,
        rollout_step: int = 0,
        data_indices: IndexCollection | None = None,
    ) -> torch.Tensor:
        """Advance a single dataset's input state for the next rollout step.

        Supports model outputs shaped like ``(B, T, E, G, V)``.
        """
        keep_steps = min(self.num_input_steps, self.num_output_steps)

        x = x.roll(-keep_steps, dims=1)

        # Compute batch indices for the output offsets of this rollout step
        output_batch_indices = self.get_batch_output_indices(rollout_step=rollout_step)

        for i in range(keep_steps):
            # Get prognostic variables
            x[:, -(i + 1), ..., data_indices.model.input.prognostic] = y_pred[
                :,
                -(i + 1),
                ...,
                data_indices.model.output.prognostic,
            ]

            batch_time_index = output_batch_indices[-(i + 1)]

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
