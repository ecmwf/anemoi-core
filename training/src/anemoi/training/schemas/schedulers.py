# (C) Copyright 2024-2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

from enum import Enum
from typing import Annotated
from typing import Literal
from typing import Union

from pydantic import Field
from pydantic import PositiveInt

from .utils import BaseModel


class ImplementedRollout(str, Enum):
    static = "anemoi.training.schedulers.rollout.Static"
    positional = "anemoi.training.schedulers.rollout.indexed.PositionalIndexed"
    positional_epoch = "anemoi.training.schedulers.rollout.indexed.EpochPositionalIndexed"
    positional_step = "anemoi.training.schedulers.rollout.indexed.StepPositionalIndexed"
    lookup = "anemoi.training.schedulers.rollout.indexed.Lookup"
    lookup_epoch = "anemoi.training.schedulers.rollout.indexed.EpochLookup"
    lookup_step = "anemoi.training.schedulers.rollout.indexed.StepLookup"
    random_list = "anemoi.training.schedulers.rollout.randomise.RandomList"
    random_range = "anemoi.training.schedulers.rollout.randomise.RandomRange"
    increasing_random = "anemoi.training.schedulers.rollout.randomise.IncreasingRandom"
    increasing_random_epoch = "anemoi.training.schedulers.rollout.randomise.EpochIncreasingRandom"
    increasing_random_step = "anemoi.training.schedulers.rollout.randomise.StepIncreasingRandom"
    stepped = "anemoi.training.schedulers.rollout.stepped.Stepped"
    stepped_epoch = "anemoi.training.schedulers.rollout.stepped.EpochStepped"
    stepped_step = "anemoi.training.schedulers.rollout.stepped.StepStepped"


class BaseRolloutSchema(BaseModel):
    target_: ImplementedRollout = Field(..., alias="_target_")
    "target rollout schema from anemoi.training.schedulers.rollout."


class InterEpochMixinSchema(BaseModel):
    adjust_maximum: Union[int, dict[int, int], dict[Literal["step", "epoch"], dict[int, int]]] = Field(example=1)
    "Adjust current maximum of rollout scheduler for inter epoch changes."


class StaticRolloutSchema(BaseRolloutSchema):
    """Static rollout configuration."""

    target_: Literal[ImplementedRollout.static] = Field(..., alias="_target_")
    rollout_value: PositiveInt = Field(example=1)
    "Static rollout value."


class PositionalIndexedSchema(BaseRolloutSchema):
    """Positional indexed based rollout."""

    target_: Literal[ImplementedRollout.positional] = Field(..., alias="_target_")
    rollouts: list[PositiveInt] = Field(example=[1, 2, 3, 4])
    "List of rollout values."
    num_times_per_element: PositiveInt = Field(example=1)
    "Number of times to remain at an element."
    step_type: Literal["step", "epoch"] = Field(example="epoch")
    "Type of step, either 'epoch' or 'step'."


class EpochPositionalIndexedSchema(PositionalIndexedSchema):
    """Positional indexed based rollout based on epoch."""

    target_: Literal[ImplementedRollout.positional_epoch] = Field(..., alias="_target_")
    step_type: Literal["epoch"] = "epoch"


class StepPositionalIndexedSchema(PositionalIndexedSchema):
    """Positional indexed based rollout based on step."""

    target_: Literal[ImplementedRollout.positional_step] = Field(..., alias="_target_")
    step_type: Literal["step"] = "step"


class LookupSchema(BaseRolloutSchema):
    """Lookup dictionary based rollout."""

    target_: Literal[ImplementedRollout.lookup] = Field(..., alias="_target_")
    table: dict[PositiveInt, PositiveInt] = Field(example={1: 1, 2: 2, 3: 3})
    "Lookup dictionary to from `step_type` to rollout value."
    step_type: Literal["step", "epoch"] = Field(example="epoch")
    "Type of step, either 'epoch' or 'step'."


class EpochLookupSchema(LookupSchema):
    """Lookup dictionary based rollout based on epoch."""

    target_: Literal[ImplementedRollout.lookup_epoch] = Field(..., alias="_target_")
    step_type: Literal["epoch"] = "epoch"


class StepLookupSchema(LookupSchema):
    """Lookup dictionary based rollout based on step."""

    target_: Literal[ImplementedRollout.lookup_step] = Field(..., alias="_target_")
    step_type: Literal["step"] = "step"


class RandomListSchema(BaseRolloutSchema):
    """Random list based rollout."""

    target_: Literal[ImplementedRollout.random_list] = Field(..., alias="_target_")
    rollouts: list[PositiveInt] = Field(example=[1, 2, 3, 4])
    "List of rollout values."


class RandomRangeSchema(BaseRolloutSchema):
    """Random sample from range based rollout."""

    target_: Literal[ImplementedRollout.random_range] = Field(..., alias="_target_")
    minimum: PositiveInt = Field(example=1)
    "Minimum rollout value."
    maximum: PositiveInt = Field(example=10)
    "Maximum rollout value."
    step: PositiveInt = Field(example=1)
    "Step of range."


class IncreasingRandomSchema(BaseRolloutSchema):
    """Increasing random range sample based rollout."""

    target_: Literal[ImplementedRollout.increasing_random] = Field(..., alias="_target_")
    minimum: PositiveInt = Field(example=1)
    "Minimum rollout value."
    maximum: int = Field(example=10)
    "Maximum rollout value."
    range_step: PositiveInt = Field(example=1)
    "Step of range."
    every_n: PositiveInt = Field(example=5)
    "Number of steps or epochs to step the rollout value."
    increment: Union[int, dict[int, int], dict[Literal["step", "epoch"], dict[int, int]]] = Field(1, example=1)
    "Value to increment the rollout by."
    step_type: Literal["step", "epoch"] = Field(example="epoch")
    "Type of step, either 'epoch' or 'step'."


class EpochIncreasingRandomSchema(IncreasingRandomSchema):
    """Increasing random range sample based rollout on epoch."""

    target_: Literal[ImplementedRollout.increasing_random_epoch] = Field(..., alias="_target_")
    step_type: Literal["epoch"] = "epoch"
    every_n: PositiveInt = Field(example=5, alias="every_n_epochs")
    "Number of epochs for when to step the rollout."


class StepIncreasingRandomSchema(IncreasingRandomSchema, InterEpochMixinSchema):
    """Increasing random range sample based rollout on step."""

    target_: Literal[ImplementedRollout.increasing_random_step] = Field(..., alias="_target_")
    step_type: Literal["step"] = "step"
    every_n: PositiveInt = Field(example=5, alias="every_n_steps")
    "Number of steps for when to step the rollout."


class SteppedSchema(BaseRolloutSchema):
    """Increasing random range sample based rollout."""

    target_: Literal[ImplementedRollout.stepped] = Field(..., alias="_target_")
    minimum: PositiveInt = Field(example=1)
    "Minimum rollout value."
    maximum: int = Field(example=10)
    "Maximum rollout value."
    every_n: PositiveInt = Field(example=5)
    "Number of steps or epochs to step the rollout value."
    increment: Union[int, dict[int, int], dict[Literal["step", "epoch"], dict[int, int]]] = Field(1, example=1)
    "Value to increment the rollout by."
    step_type: Literal["step", "epoch"] = Field(example="epoch")
    "Type of step, either 'epoch' or 'step'."


class EpochSteppedSchema(SteppedSchema):
    """Increasing random range sample based rollout on epoch."""

    target_: Literal[ImplementedRollout.stepped_epoch] = Field(..., alias="_target_")
    step_type: Literal["epoch"] = "epoch"
    every_n: PositiveInt = Field(example=5, alias="every_n_epochs")
    "Number of epochs for when to step the rollout."


class StepSteppedSchema(SteppedSchema, InterEpochMixinSchema):
    """Increasing random range sample based rollout on step."""

    target_: Literal[ImplementedRollout.stepped_step] = Field(..., alias="_target_")
    step_type: Literal["step"] = "step"
    every_n: PositiveInt = Field(example=5, alias="every_n_steps")
    "Number of steps for when to step the rollout."


RolloutSchemas = Annotated[
    Union[
        StaticRolloutSchema,
        PositionalIndexedSchema,
        EpochPositionalIndexedSchema,
        StepPositionalIndexedSchema,
        LookupSchema,
        EpochLookupSchema,
        StepLookupSchema,
        RandomListSchema,
        RandomRangeSchema,
        IncreasingRandomSchema,
        EpochIncreasingRandomSchema,
        StepIncreasingRandomSchema,
        SteppedSchema,
        EpochSteppedSchema,
        StepSteppedSchema,
    ],
    Field(discriminator="target_"),
]
"Union of all rollout schemas."
