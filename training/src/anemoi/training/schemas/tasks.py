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
from pydantic import NonNegativeFloat
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
    validation_rollout: PositiveInt | None = Field(default=None, example=[None, 6, 12])
    "Minimum number of rollout steps unrolled in validation; the training rollout is always unrolled. "
    "The val loss is averaged over the training rollout only."
    exact_validation_rollout: bool = Field(default=False)
    "Unroll exactly ``validation_rollout`` steps in validation, also when the training rollout is longer. "
    "The val loss then averages over those steps only."


class DAForecasterSchema(BaseModel):
    """Configuration for the data-assimilation forecasting task."""

    target_: Literal["anemoi.training.tasks.DAForecaster"] = Field(..., alias="_target_")
    "Task class path for the data-assimilation forecasting task."
    multistep_input: PositiveInt = Field(example=2)
    "Number of input timesteps provided to the model."
    multistep_output: PositiveInt = Field(example=1)
    "Number of output timesteps the model should predict."
    timestep: str = Field(example="6H")
    "Timestep string (e.g. '6H') defining the frequency of the input and output steps."
    rollout: RolloutSchema = Field(...)
    "Rollout configuration of the forecast steps (after the DA cycles)."
    validation_rollout: PositiveInt | None = Field(default=None, example=[None, 6, 12])
    "Minimum number of forecast steps unrolled in validation; the training rollout is always unrolled."
    exact_validation_rollout: bool = Field(default=False)
    "Unroll exactly ``validation_rollout`` steps in validation, also when the training rollout is longer. "
    "The val loss then averages over those steps only."
    da_cycles: NonNegativeInt = Field(default=0)
    "Number of data-assimilation cycles run before the forecast rollout."
    da_loss_weight: NonNegativeFloat = Field(default=0.0)
    "Weight of each DA cycle's loss (0 skips it). The loss is averaged over the forecast steps."
    da_flow_dependent_skip: bool = Field(default=False)
    "Use the pre-observation model background as the residual base after a DA blend (needs skip_input support)."
    checkpoint_steps: NonNegativeInt = Field(default=0)
    "Number of leading training steps (DA cycles first) whose model call is activation-checkpointed."


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
    validation_rollout: PositiveInt | None = Field(default=None, example=[None, 6, 12])
    "Minimum number of rollout steps unrolled in validation; the training rollout is always unrolled. "
    "The val loss is averaged over the training rollout only."
    exact_validation_rollout: bool = Field(default=False)
    "Unroll exactly ``validation_rollout`` steps in validation, also when the training rollout is longer. "
    "The val loss then averages over those steps only."


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
    ForecasterSchema | DAForecasterSchema | OffsetForecasterSchema | AutoencoderTaskSchema | TemporalDownscalerSchema,
    Discriminator("target_"),
]
