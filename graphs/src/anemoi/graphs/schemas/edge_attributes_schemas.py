# (C) Copyright 2024-2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from enum import Enum
from typing import Literal

from pydantic import Field

from anemoi.utils.schemas import BaseModel


class ImplementedEdgeAttributeSchema(str, Enum):
    edge_length = "anemoi.graphs.edges.attributes.EdgeLength"
    edge_dirs = "anemoi.graphs.edges.attributes.EdgeDirection"
    azimuth = "anemoi.graphs.edges.attributes.Azimuth"
    gaussian_weights = "anemoi.graphs.edges.attributes.GaussianDistanceWeights"


class BaseEdgeAttributeSchema(BaseModel):
    target_: ImplementedEdgeAttributeSchema = Field(..., alias="_target_")
    "Edge attribute builder object from anemoi.graphs.edges.attributes"
    norm: Literal["unit-max", "l1", "l2", "unit-sum", "unit-std"] = Field(example="unit-std")
    "Normalisation method applied to the edge attribute."


class EdgeAttributeFromNodeSchema(BaseModel):
    target_: Literal[
        "anemoi.graphs.edges.attributes.AttributeFromSourceNode",
        "anemoi.graphs.edges.attributes.AttributeFromTargetNode",
    ] = Field(..., alias="_target_")
    "Edge attributes from node attribute"
    norm: Literal["unit-max", "l1", "l2", "unit-sum", "unit-std"] = Field(example="unit-std")
    "Normalisation method applied to the edge attribute."


EdgeAttributeSchema = BaseEdgeAttributeSchema | EdgeAttributeFromNodeSchema
