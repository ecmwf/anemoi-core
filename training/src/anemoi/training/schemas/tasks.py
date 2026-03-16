# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Literal

from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveInt
from anemoi.utils.schemas import BaseModel


class ForecastingTaskSchema(BaseModel):
    """Configuration for forecasting tasks."""

    target_: Literal["anemoi.training.tasks.ForecastingTask"] = Field(..., alias="_target_")
    "Task class path for the forecasting task."
    multistep_input: list[str] = Field(example=[2])
    "Number of input timesteps provided to the model."
    multistep_output: list[str] = Field(example=[1])
    "Number of output timesteps the model should predict."
    timestep: str = Field(example="6H")
    "Timestep string (e.g. '6H') defining the frequency of the input and output steps."
    rollout_start: NonNegativeInt = Field(1, example=1)
    "Initial number of rollout steps for training."
    rollout_epoch_increment: NonNegativeInt = Field(0, example=0)
    "Number of rollout steps to add at the end of each epoch."
    rollout_max: PositiveInt = Field(1, example=1)
    "Maximum number of rollout steps for training."


class AutoencodingTaskSchema(BaseModel):
    """Configuration for autoencoding tasks."""

    target_: Literal["anemoi.training.tasks.AutoencodingTask"] = Field(..., alias="_target_")
    "Task class path for the autoencoding task."


class DownscalingTaskSchema(BaseModel):
    """Configuration for downscaling tasks."""

    target_: Literal["anemoi.training.tasks.DownscalingTask"] = Field(..., alias="_target_")
    "Task class path for the downscaling task."


class TimeInterpolationTaskSchema(BaseModel):
    """Configuration for time interpolation tasks."""

    target_: Literal["anemoi.training.tasks.TimeInterpolationTask"] = Field(..., alias="_target_")
    "Task class path for the time interpolation task."
    inputs_offsets: list[str] = Field(example=["0H", "6H"])
    "List of input timestep offsets as duration strings (e.g. ['0H', '6H'])."
    outputs_offsets: list[str] = Field(example=["1H", "2H", "3H", "4H", "5H"])
    "List of output timestep offsets as duration strings (e.g. ['1H', '2H', '3H', '4H', '5H'])."


TaskSchema = ForecastingTaskSchema | AutoencodingTaskSchema | DownscalingTaskSchema | TimeInterpolationTaskSchema
