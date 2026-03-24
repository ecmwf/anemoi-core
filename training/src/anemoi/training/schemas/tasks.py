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
from pydantic import PositiveInt

from anemoi.utils.schemas import BaseModel


class TaskRollout(BaseModel):
    """Rollout configuration for task."""

    start: NonNegativeInt = Field(default=1, example=1)
    "Number of rollouts to start with."
    epoch_increment: NonNegativeInt = Field(default=0, example=0)
    "Number of epochs to increment the rollout."
    max: NonNegativeInt = Field(default=1, example=1)
    "Maximum number of rollouts."


class ForecastingTaskSchema(BaseModel):
    """Configuration for forecasting tasks."""

    target_: Literal["anemoi.training.tasks.ForecastingTask"] = Field(..., alias="_target_")
    "Task class path for the forecasting task."
    multistep_input: PositiveInt = Field(example=2)
    "Number of input timesteps provided to the model."
    multistep_output: PositiveInt = Field(example=1)
    "Number of output timesteps the model should predict."
    timestep: str = Field(example="6H")
    "Timestep string (e.g. '6H') defining the frequency of the input and output steps."
    rollout: TaskRollout = Field(default_factory=TaskRollout)
    "Rollout configuration for autoregressive training."


class AutoencodingTaskSchema(BaseModel):
    """Configuration for autoencoding tasks."""

    target_: Literal["anemoi.training.tasks.AutoencodingTask"] = Field(..., alias="_target_")
    "Task class path for the autoencoding task."


class DownscalingTaskSchema(BaseModel):
    """Configuration for downscaling tasks."""

    target_: Literal["anemoi.training.tasks.SpatialDownscalingTask"] = Field(..., alias="_target_")
    "Task class path for the downscaling task."


class TimeInterpolationTaskSchema(BaseModel):
    """Configuration for time interpolation tasks."""

    target_: Literal["anemoi.training.tasks.TemporalDownscalingTask"] = Field(..., alias="_target_")
    "Task class path for the time interpolation task."
    input_timestep: str = Field(example="6H")
    "Input data timestep as a duration string (e.g. '6H')."
    output_timestep: str = Field(example="1H")
    "Desired output timestep as a duration string (e.g. '1H')."
    output_left_boundary: bool = Field(default=False, example=False)
    "Whether to include the left boundary in the output."
    output_right_boundary: bool = Field(default=False, example=False)
    "Whether to include the right boundary in the output."


TaskSchema = Annotated[
    ForecastingTaskSchema | AutoencodingTaskSchema | DownscalingTaskSchema | TimeInterpolationTaskSchema,
    Discriminator("target_"),
]
