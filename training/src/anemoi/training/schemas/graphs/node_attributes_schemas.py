# (C) Copyright 2024-2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from __future__ import annotations

import logging
from typing import Literal
from typing import Union

from pydantic import BaseModel
from pydantic import Field

LOGGER = logging.getLogger(__name__)


class PlanarAreaWeightSchema(BaseModel):
    target_: Literal[
        "anemoi.graphs.nodes.attributes.AreaWeights",
        "anemoi.graphs.nodes.attributes.PlanarAreaWeights",
        "anemoi.graphs.nodes.attributes.UniformWeights",
    ] = Field(..., alias="_target_")
    "Implementation of the area of the nodes as the weights from anemoi.graphs.nodes.attributes."
    norm: Literal["unit-max", "l1", "l2", "unit-sum", "unit-std"] = Field(example="unit-max")
    "Normalisation of the weights."


class SphericalAreaWeightSchema(BaseModel):
    target_: Literal["anemoi.graphs.nodes.attributes.SphericalAreaWeights"] = Field(..., alias="_target_")
    "Implementation of the 3D area of the nodes as the weights from anemoi.graphs.nades.attributes."
    norm: Literal["unit-max", "l1", "l2", "unit-sum", "unit-std"] = Field(example="unit-max")
    "Normalisation of the weights."
    fill_value: float = Field(example=0)
    "Value to fill the empty regions."


class CutOutMaskSchema(BaseModel):
    target_: Literal["anemoi.graphs.nodes.attributes.CutOutMask"] = Field(..., alias="_target_")
    "Implementation of the cutout mask from anemoi.graphs.nodes.attributes."


class NonmissingZarrVariableSchema(BaseModel):
    target_: Literal["anemoi.graphs.nodes.attributes.NonmissingZarrVariable"] = Field(..., alias="_target_")
    (
        "Implementation of a mask from the nonmissing values of a anemoi-datasets variable "
        "from anemoi.graphs.nodes.attributes."
    )
    variable: str
    "The anemoi-datasets variable to use."


class BooleanOperationSchema(BaseModel):
    target_: Literal[
        "anemoi.graphs.nodes.attributes.BooleanNot",
        "anemoi.graphs.nodes.attributes.BooleanAndMask",
        "anemoi.graphs.nodes.attributes.BooleanOrMask",
    ] = Field(..., alias="_target_")
    "Implementation of boolean masks from anemoi.graphs.nodes.attributes"


class DistanceFromPointSchema(BaseModel):
    target_: Literal["anemoi.graphs.nodes.attributes.DistanceFromPointAttribute",] = Field(..., alias="_target_")
    norm: Literal["unit-max", "l1", "l2", "unit-sum", "unit-std"] = Field(example="unit-max")
    "Normalisation of the weights."
    dst_point: str | tuple[float, float] = Field(default="lam_center")
    "Coordinates of the point to compute distance from. /"
    "If 'lam_center' is passed, distane from lam center point will be computed"

    "Implementation of distance from point from anemoi.graphs.nodes.attributes"


NodeAttributeSchemas = Union[
    PlanarAreaWeightSchema,
    SphericalAreaWeightSchema,
    CutOutMaskSchema,
    NonmissingZarrVariableSchema,
    BooleanOperationSchema,
    DistanceFromPointSchema,
]
