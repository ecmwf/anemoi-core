# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from typing import Annotated
from typing import Literal
from typing import Union

from pydantic import Field
from pydantic import model_validator

from anemoi.utils.schemas import BaseModel


class ReluBoundingSchema(BaseModel):
    target_: Literal["anemoi.models.layers.bounding.ReluBounding"] = Field(..., alias="_target_")
    "Relu bounding object defined in anemoi.models.layers.bounding."
    variables: list[str]
    "List of variables to bound using the Relu method."


class LeakyReluBoundingSchema(ReluBoundingSchema):
    target_: Literal["anemoi.models.layers.bounding.LeakyReluBounding"] = Field(..., alias="_target_")
    "Leaky Relu bounding object defined in anemoi.models.layers.bounding."


class FractionBoundingSchema(BaseModel):
    target_: Literal["anemoi.models.layers.bounding.FractionBounding"] = Field(..., alias="_target_")
    "Fraction bounding object defined in anemoi.models.layers.bounding."
    variables: list[str]
    "List of variables to bound using the hard tanh fraction method."
    min_val: float
    "The minimum value for the HardTanh activation. Correspond to the minimum fraction of the total_var."
    max_val: float
    "The maximum value for the HardTanh activation. Correspond to the maximum fraction of the total_var."
    total_var: str
    "Variable from which the secondary variables are derived. \
    For example, convective precipitation should be a fraction of total precipitation."


class LeakyFractionBoundingSchema(FractionBoundingSchema):
    target_: Literal["anemoi.models.layers.bounding.LeakyFractionBounding"] = Field(..., alias="_target_")
    "Leaky fraction bounding object defined in anemoi.models.layers.bounding."


class HardtanhBoundingSchema(BaseModel):
    target_: Literal["anemoi.models.layers.bounding.HardtanhBounding"] = Field(..., alias="_target_")
    "Hard tanh bounding method function from anemoi.models.layers.bounding."
    variables: list[str]
    "List of variables to bound using the hard tanh method."
    min_val: float
    "The minimum value for the HardTanh activation."
    max_val: float
    "The maximum value for the HardTanh activation."


class LeakyHardtanhBoundingSchema(HardtanhBoundingSchema):
    target_: Literal["anemoi.models.layers.bounding.LeakyHardtanhBounding"] = Field(..., alias="_target_")
    "Leaky hard tanh bounding method function from anemoi.models.layers.bounding."


class NormalizedReluBoundingSchema(BaseModel):
    target_: Literal["anemoi.models.layers.bounding.NormalizedReluBounding"] = Field(..., alias="_target_")
    variables: list[str]
    min_val: list[float]
    normalizer: list[str]

    @model_validator(mode="after")
    def check_num_normalizers_and_min_val_matches_num_variables(
        self,
    ) -> NormalizedReluBoundingSchema:
        error_msg = f"""{self.__class__} requires that number of normalizers ({len(self.normalizer)}) or
        match the number of variables ({len(self.variables)})"""
        assert len(self.normalizer) == len(self.variables), error_msg
        error_msg = f"""{self.__class__} requires that number of min_val ({len(self.min_val)}) or  match
        the number of variables ({len(self.variables)})"""
        assert len(self.min_val) == len(self.variables), error_msg
        return self


class NormalizedLeakyReluBoundingSchema(NormalizedReluBoundingSchema):
    target_: Literal["anemoi.models.layers.bounding.NormalizedLeakyReluBounding"] = Field(..., alias="_target_")
    "Leaky normalized Relu bounding object defined in anemoi.models.layers.bounding."


BoundingSchema = Annotated[
    Union[
        ReluBoundingSchema,
        LeakyReluBoundingSchema,
        FractionBoundingSchema,
        LeakyFractionBoundingSchema,
        HardtanhBoundingSchema,
        LeakyHardtanhBoundingSchema,
        NormalizedReluBoundingSchema,
        NormalizedLeakyReluBoundingSchema,
    ],
    Field(discriminator="target_"),
]
