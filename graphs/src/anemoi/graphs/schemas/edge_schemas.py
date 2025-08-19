# (C) Copyright 2024-2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from typing import Annotated
from typing import Literal

from pydantic import Field
from pydantic import PositiveFloat
from pydantic import PositiveInt

from anemoi.utils.schemas import BaseModel


class KNNEdgeSchema(BaseModel):
    target_: Literal["anemoi.graphs.edges.KNNEdges", "anemoi.graphs.edges.ReversedKNNEdges"] = Field(
        ..., alias="_target_"
    )
    "KNN based edges implementation from anemoi.graphs.edges."
    num_nearest_neighbours: PositiveInt = Field(example=3)
    "Number of nearest neighbours. Default to 3."
    source_mask_attr_name: str | None = Field(default=None, examples=["boundary_mask"])
    "Mask to apply to source nodes of the edges. Default to None."
    target_mask_attr_name: str | None = Field(default=None, examples=["boundary_mask"])
    "Mask to apply to target nodes of the edges. Default to None."


class CutoffEdgeSchema(BaseModel):
    target_: Literal["anemoi.graphs.edges.CutOffEdges", "anemoi.graphs.edges.ReversedCutOffEdges"] = Field(
        ..., alias="_target_"
    )
    "Cut-off based edges implementation from anemoi.graphs.edges."
    cutoff_factor: PositiveFloat = Field(example=0.6)
    "Factor to multiply the grid reference distance to get the cut-off radius. Default to 0.6."
    source_mask_attr_name: str | None = Field(default=None, examples=["boundary_mask"])
    "Mask to apply to source nodes of the edges. Default to None."
    target_mask_attr_name: str | None = Field(default=None, examples=["boundary_mask"])
    "Mask to apply to target nodes of the edges. Default to None."


class MultiScaleEdgeSchema(BaseModel):
    target_: Literal["anemoi.graphs.edges.MultiScaleEdges"] = Field(
        "anemoi.graphs.edges.MultiScaleEdges",
        alias="_target_",
    )
    "Multi-casle edges implementation from anemoi.graphs.edges."
    x_hops: PositiveInt = Field(example=1)
    "Number of hops (in the refined icosahedron) between two nodes to connect them with an edge. Default to 1."
    scale_resolutions: PositiveInt | list[PositiveInt] | None = Field(examples=[1, 2, 3, 4, 5])
    "Specifies the resolution scales for computing the hop neighbourhood."
    source_mask_attr_name: str | None = Field(default=None, examples=["boundary_mask"])
    "Mask to apply to source nodes of the edges. Default to None."
    target_mask_attr_name: str | None = Field(default=None, examples=["boundary_mask"])
    "Mask to apply to target nodes of the edges. Default to None."


class ICONTopologicalEdgeSchema(BaseModel):
    target_: Literal[
        "anemoi.graphs.edges.ICONTopologicalProcessorEdges",
        "anemoi.graphs.edges.ICONTopologicalEncoderEdges",
        "anemoi.graphs.edges.ICONTopologicalDecoderEdges",
    ] = Field("anemoi.graphs.edges.ICONTopologicalProcessorEdges", alias="_target_")
    icon_mesh: str
    "The name of the ICON mesh (defines both the processor mesh and the data)."
    source_mask_attr_name: str | None = Field(default=None, examples=["boundary_mask"])
    "Mask to apply to source nodes of the edges. Default to None."
    target_mask_attr_name: str | None = Field(default=None, examples=["boundary_mask"])
    "Mask to apply to target nodes of the edges. Default to None."


class EdgeAttributeSchema(BaseModel):
    target_: Literal["anemoi.graphs.edges.attributes.EdgeLength", "anemoi.graphs.edges.attributes.EdgeDirection"] = (
        Field("anemoi.graphs.edges.attributes.EdgeLength", alias="_target_")
    )
    "Edge attributes object from anemoi.graphs.edges."
    norm: Literal["unit-max", "l1", "l2", "unit-sum", "unit-std"] = Field(example="unit-std")
    "Normalisation method applied to the edge attribute."


EdgeBuilderSchemas = Annotated[
    KNNEdgeSchema | CutoffEdgeSchema | MultiScaleEdgeSchema | ICONTopologicalEdgeSchema,
    Field(discriminator="target_"),
]
