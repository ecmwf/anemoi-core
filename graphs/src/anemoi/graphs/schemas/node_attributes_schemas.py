# (C) Copyright 2024-2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import logging
from typing import Literal

from pydantic import Field

from anemoi.utils.schemas import BaseModel

LOGGER = logging.getLogger(__name__)


class PlanarAreaWeightSchema(BaseModel):
    target_: Literal[
        "anemoi.graphs.nodes.attributes.PlanarAreaWeights",
        "anemoi.graphs.nodes.attributes.UniformWeights",
        "anemoi.graphs.nodes.attributes.CosineLatWeightedAttribute",
        "anemoi.graphs.nodes.attributes.IsolatitudeAreaWeights",
    ] = Field(..., alias="_target_")
    "Implementation of the area of the nodes as the weights from anemoi.graphs.nodes.attributes."
    norm: Literal["unit-max", "l1", "l2", "unit-sum", "unit-std"] = Field(example="unit-max")
    "Normalisation of the weights."


class MaskedPlanarAreaWeightsSchema(BaseModel):
    target_: Literal["anemoi.graphs.nodes.attributes.MaskedPlanarAreaWeights"] = Field(..., alias="_target_")
    "Implementation of the area of the nodes as the weights from anemoi.graphs.nodes.attributes."
    mask_node_attr_name: str = Field(examples="cutout_mask")
    "Attribute name to mask the area weights."
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
    target_: Literal["anemoi.graphs.nodes.attributes.CutOutMask", "anemoi.graphs.nodes.attributes.LimitedAreaMask"] = (
        Field(..., alias="_target_")
    )
    "Implementation of the area masks from anemoi.graphs.nodes.attributes."


class GridsMaskSchema(BaseModel):
    target_: Literal["anemoi.graphs.nodes.attributes.GridsMask"] = Field(..., alias="_target_")
    "Implementation of the grids mask from anemoi.graphs.nodes.attributes."
    grids: list[int] | int = Field(examples=[0, [0]])
    "Position of the grids to consider as True."


class NonmissingAnemoiDatasetVariableSchema(BaseModel):
    target_: Literal["anemoi.graphs.nodes.attributes.NonmissingAnemoiDatasetVariable"] = Field(..., alias="_target_")
    (
        "Implementation of a mask from the nonmissing values of a anemoi-datasets variable "
        "from anemoi.graphs.nodes.attributes."
    )
    variable: str
    "The anemoi-datasets variable to use."


SingleAttributeSchema = (
    PlanarAreaWeightSchema
    | MaskedPlanarAreaWeightsSchema
    | SphericalAreaWeightSchema
    | CutOutMaskSchema
    | GridsMaskSchema
    | NonmissingAnemoiDatasetVariableSchema
)


class BooleanOperationSchema(BaseModel):
    target_: Literal[
        "anemoi.graphs.nodes.attributes.BooleanNot",
        "anemoi.graphs.nodes.attributes.BooleanAndMask",
        "anemoi.graphs.nodes.attributes.BooleanOrMask",
    ] = Field(..., alias="_target_")
    "Implementation of boolean masks from anemoi.graphs.nodes.attributes"
    masks: str | SingleAttributeSchema | list[str | SingleAttributeSchema]


NodeAttributeSchemas = SingleAttributeSchema | BooleanOperationSchema
