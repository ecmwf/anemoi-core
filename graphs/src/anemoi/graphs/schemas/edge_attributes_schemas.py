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
    directional_harmonics = "anemoi.graphs.edges.attributes.DirectionalHarmonics"
    azimuth = "anemoi.graphs.edges.attributes.Azimuth"
    gaussian_weights = "anemoi.graphs.edges.attributes.GaussianDistanceWeights"
    radial_basis_features = "anemoi.graphs.edges.attributes.RadialBasisFeatures"


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


class DirectionalHarmonicsSchema(BaseModel):
    target_: Literal["anemoi.graphs.edges.attributes.DirectionalHarmonics"] = Field(..., alias="_target_")
    "Directional harmonics from edge directions"
    order: int = Field(default=3, description="Maximum order of harmonics to compute")
    norm: Literal["unit-max", "l1", "l2", "unit-sum", "unit-std"] | None = Field(
        default=None, description="Normalization method"
    )
    dtype: str = Field(default="float32", description="Data type for computations")


class RadialBasisFeaturesSchema(BaseModel):
    target_: Literal["anemoi.graphs.edges.attributes.RadialBasisFeatures"] = Field(..., alias="_target_")
    "Radial basis function features from edge distances"
    r_scale: float | None = Field(default=None, description="Global scale factor (None for adaptive per-node scaling)")
    centers: list[float] | None = Field(default=None, description="RBF center positions [0, 1]")
    sigma: float = Field(default=0.2, description="Width of Gaussian RBF functions")
    epsilon: float = Field(default=1e-10, description="Small constant to avoid division by zero")
    dtype: str = Field(default="float32", description="Data type for computations")


EdgeAttributeSchema = (
    BaseEdgeAttributeSchema | EdgeAttributeFromNodeSchema | DirectionalHarmonicsSchema | RadialBasisFeaturesSchema
)
