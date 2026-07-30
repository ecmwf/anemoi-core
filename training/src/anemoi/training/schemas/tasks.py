# (C) Copyright 2026 Anemoi contributors.
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


class ForecasterSchema(BaseModel):
    """Configuration for forecasting tasks."""

    target_: Literal["anemoi.training.tasks.Forecaster"] = Field(..., alias="_target_")
    "Task class path for the forecasting task."
    multistep_input: PositiveInt | None = Field(default=None, example=2)
    "Legacy: number of input timesteps provided to the model."
    multistep_output: PositiveInt | None = Field(default=None, example=1)
    "Legacy: number of output timesteps the model should predict."
    timestep: str | None = Field(default=None, example="6H")
    "Legacy: timestep string (e.g. '6H') defining the frequency of the input and output steps."
    input_offsets: list[str] | None = Field(default=None, example=["-6H", "0H"])
    "Offset-based: input time offsets as duration strings (e.g. ['-6H', '0H'])."
    output_offsets: list[str] | None = Field(default=None, example=["6H"])
    "Offset-based: output time offsets as duration strings (e.g. ['6H'])."
    rollout_shift: str | None = Field(default="0H", example="6H")
    "Offset-based: time shift applied to the offsets between rollout steps (e.g. '6H')."
    rollout: RolloutSchema = Field(...)
    "Rollout configuration for autoregressive training."
    validation_rollout: NonNegativeInt | None = Field(default=None, example=[None, 6, 12])
    "Number of rollouts to use for validation. If unset, validation uses the training rollout."


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
