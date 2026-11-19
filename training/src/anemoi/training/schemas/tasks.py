# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Annotated
from typing import Literal

from pydantic import Discriminator
from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveFloat
from pydantic import PositiveInt

from anemoi.utils.schemas import BaseModel


class RolloutSchema(BaseModel):
    """Rollout configuration for task."""

    start: NonNegativeInt = Field(example=1)
    "Number of rollouts to start with."
    epoch_increment: NonNegativeInt = Field(example=0)
    "Number of epochs to increment the rollout."
    maximum: NonNegativeInt = Field(example=1)
    "Maximum number of rollouts."


class PushforwardSchema(BaseModel):
    """Pushforward training rollouts: only the first and last steps get gradients and a loss."""

    lengths: list[PositiveInt] = Field(example=[12, 24, 36, 48, 60])
    "Rollout lengths; each training batch draws one (the same on all ranks)."
    anchor_steps: NonNegativeInt = Field(default=0)
    "Number of first steps trained with gradients and loss."
    grad_steps: PositiveInt = Field(default=1)
    "Number of last steps trained with gradients and loss. The steps in between run without gradients."


class ForecasterSchema(BaseModel):
    """Configuration for forecasting tasks."""

    target_: Literal["anemoi.training.tasks.Forecaster"] = Field(..., alias="_target_")
    "Task class path for the forecasting task."
    multistep_input: PositiveInt = Field(example=2)
    "Number of input timesteps provided to the model."
    multistep_output: PositiveInt = Field(example=1)
    "Number of output timesteps the model should predict."
    timestep: str = Field(example="6H")
    "Timestep string (e.g. '6H') defining the frequency of the input and output steps."
    rollout: RolloutSchema = Field(...)
    "Rollout configuration for autoregressive training."
    validation_rollout: NonNegativeInt | None = Field(default=None, example=[None, 6, 12])
    "Number of rollouts to use for validation. If unset, validation uses the training rollout."
    rollout_loss_weights: list[PositiveFloat] | None = Field(default=None, example=[None, [1.0, 0.8, 0.7]])
    "Relative training-loss weight of each rollout step (index 0 = first step), renormalised over the current "
    "rollout window. Needs an entry for every training step. Validation stays unweighted."
    log_rollout_step_losses: bool = Field(default=False)
    "Log the unweighted training loss of every rollout step (epoch mean)."
    pushforward: PushforwardSchema | None = Field(default=None)
    "Pushforward training rollouts (replaces the rollout schedule in training)."


class AutoencoderTaskSchema(BaseModel):
    """Configuration for autoencoding tasks."""

    target_: Literal["anemoi.training.tasks.Autoencoder"] = Field(..., alias="_target_")
    "Task class path for the autoencoding task."


class TemporalDownscalerSchema(BaseModel):
    """Configuration for temporal downscaling task."""

    target_: Literal["anemoi.training.tasks.TemporalDownscaler"] = Field(..., alias="_target_")
    "Task class path for the temporal downscaling task."
    input_timestep: str = Field(example="6H")
    "Input data timestep as a duration string (e.g. '6H')."
    output_timestep: str = Field(example="1H")
    "Desired output timestep as a duration string (e.g. '1H')."
    output_left_boundary: bool = Field(example=False)
    "Whether to include the left boundary in the output."
    output_right_boundary: bool = Field(example=False)
    "Whether to include the right boundary in the output."


TaskSchema = Annotated[
    ForecasterSchema | AutoencoderTaskSchema | TemporalDownscalerSchema,
    Discriminator("target_"),
]
