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
from pydantic import model_validator

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
    cutoff_factor: PositiveFloat | None = Field(default=None, example=0.6)
    "Factor to multiply the grid reference distance to get the cut-off radius. Mutually exclusive with cutoff_distance_km."
    cutoff_distance_km: PositiveFloat | None = Field(default=None, example=500.0)
    "Cutoff radius in kilometers. Mutually exclusive with cutoff_factor."
    source_mask_attr_name: str | None = Field(default=None, examples=["boundary_mask"])
    "Mask to apply to source nodes of the edges. Default to None."
    target_mask_attr_name: str | None = Field(default=None, examples=["boundary_mask"])
    "Mask to apply to target nodes of the edges. Default to None."
    max_num_neighbours: PositiveInt = Field(default=64, example=64)
    "Maximum number of nearest neighbours to consider when building edges. Default to 64."

    @model_validator(mode="after")
    def validate_cutoff_params(self):
        """Validate that exactly one of cutoff_factor or cutoff_distance_km is provided."""
        cutoff_factor = self.cutoff_factor
        cutoff_distance_km = self.cutoff_distance_km

        if cutoff_factor is None and cutoff_distance_km is None:
            raise ValueError("Either cutoff_factor or cutoff_distance_km must be provided.")
        if cutoff_factor is not None and cutoff_distance_km is not None:
            raise ValueError("cutoff_factor and cutoff_distance_km are mutually exclusive. Provide only one.")

        return self


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
