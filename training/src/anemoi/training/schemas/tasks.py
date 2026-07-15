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
from pydantic import model_validator

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


class SpatialDownscalerSchema(BaseModel):
    """Configuration for spatial residual downscaling."""

    target_: Literal["anemoi.training.tasks.SpatialDownscaler"] = Field(..., alias="_target_")
    "Task class path for the spatial downscaling task."
    input_datasets: list[str] = Field(..., min_length=1)
    "Datasets supplied as spatially coarse/source inputs."
    output_datasets: list[str] = Field(..., min_length=1)
    "Datasets predicted as spatially fine/target outputs."
    input_offsets: list[int] = Field(..., min_length=1)
    "Dataset-relative integer offsets of the source state(s)."
    output_offsets: list[int] = Field(..., min_length=1)
    "Dataset-relative integer offsets of the target state(s). Must be a subset of `input_offsets`."

    @model_validator(mode="after")
    def _validate_offsets(self) -> "SpatialDownscalerSchema":
        if len(set(self.input_offsets)) != len(self.input_offsets):
            raise ValueError(f"input_offsets contains duplicates: {self.input_offsets}")
        if len(set(self.output_offsets)) != len(self.output_offsets):
            raise ValueError(f"output_offsets contains duplicates: {self.output_offsets}")
        missing = sorted(set(self.output_offsets) - set(self.input_offsets))
        if missing:
            raise ValueError(
                f"output_offsets {missing} have no matching input_offsets; "
                f"output_offsets must be a subset of input_offsets ({sorted(set(self.input_offsets))}).",
            )
        return self


TaskSchema = Annotated[
    ForecasterSchema | AutoencoderTaskSchema | TemporalDownscalerSchema | SpatialDownscalerSchema,
    Discriminator("target_"),
]
