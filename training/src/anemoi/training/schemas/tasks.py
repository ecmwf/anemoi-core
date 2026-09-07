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
    """Configuration for multistep forecasting tasks."""

    target_: Literal["anemoi.training.tasks.Forecaster"] = Field(..., alias="_target_")
    "Task class path for the multistep forecasting task."
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


class OffsetForecasterSchema(BaseModel):
    """Configuration for the offset-based forecasting task."""

    target_: Literal["anemoi.training.tasks.OffsetForecaster"] = Field(..., alias="_target_")
    "Task class path for the offset-based forecasting task."
    input_offsets: list[str] = Field(example=["-6H", "0H"], min_length=1)
    "Input time offsets as duration strings."
    output_offsets: list[str] = Field(example=["6H"], min_length=1)
    "Output time offsets as duration strings."
    rollout_shift: str = Field(default="default", example="6H")
    "Time shift applied to the offsets between rollout steps. 'default' infers the largest valid shift."
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


class QueryForecastingSchema(BaseModel):
    """Configuration for query-first direct forecasting."""

    target_: Literal["anemoi.training.tasks.QueryForecasting"] = Field(
        ...,
        alias="_target_",
    )
    lead_times: list[str] = Field(min_length=1)
    input_history: str
    samples_per_epoch: PositiveInt
    reference_provenance: str
    target_variables: list[str] | None = None
    input_variables: list[str] | None = None
    source_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    field_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    history_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    max_input_times: PositiveInt = 4
    input_context_margin_degrees: float = Field(default=0.0, ge=0.0)
    global_context_sources: list[str] = Field(default_factory=list)
    target_regions: list[list[float]] = Field(default_factory=list)
    target_regions_by_provenance: dict[str, list[list[float]]] = Field(
        default_factory=dict,
    )
    variable_weights: dict[str, float] = Field(default_factory=dict)
    provenance_weights: dict[str, float] = Field(default_factory=dict)
    loss_weights: dict[str, float] = Field(default_factory=dict)
    spatial_weighting: Literal["uniform", "cosine_latitude"] = "uniform"
    aliases: dict[str, str] = Field(default_factory=dict)
    availability_policy: Literal["retrospective"] = "retrospective"
    availability_lag: dict[str, str] = Field(default_factory=dict)
    validation_seed: int = 17
    validation_samples: PositiveInt = 16
    seed: int = 42


TaskSchema = Annotated[
    ForecasterSchema
    | OffsetForecasterSchema
    | AutoencoderTaskSchema
    | TemporalDownscalerSchema
    | QueryForecastingSchema,
    Discriminator("target_"),
]
