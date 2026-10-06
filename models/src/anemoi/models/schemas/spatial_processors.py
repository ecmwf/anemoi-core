# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import Literal
from typing import Optional
from typing import Union

from pydantic import Field
from pydantic import model_validator

from anemoi.utils.schemas import BaseModel


class CrossGridProjectorSchema(BaseModel):
    """Schema for CrossGridProjector spatial preprocessor.

    Exactly one of ``file_path`` or ``edges_name`` must be provided.  The
    ``graph`` argument is injected at runtime by ``AnemoiModelInterface`` and
    is therefore not part of the config schema.
    """

    target_: Literal["anemoi.models.preprocessing.cross_grid_projector.CrossGridProjector"] = Field(
        ..., alias="_target_"
    )
    "CrossGridProjector class path."
    edges_name: Optional[tuple[str, str, str]] = Field(default=None)
    "Edge type key ``(src_node_type, relation, dst_node_type)`` in the graph. Required when not using file_path."
    edge_weight_attribute: Optional[str] = Field(default=None)
    "Edge attribute to use as interpolation weights."
    src_node_weight_attribute: Optional[str] = Field(default=None)
    "Source-node attribute to multiply into edge weights."
    file_path: Optional[str] = Field(default=None)
    "Path to a pre-computed ``.npz`` sparse projection matrix. Alternative to ``edges_name``."
    row_normalize: bool = Field(default=True)
    "If ``True``, each row of the projection matrix is normalised to sum to 1."
    autocast: bool = Field(default=False)
    "Whether to run the sparse matmul under automatic mixed precision."

    @model_validator(mode="after")
    def check_source_provided(self) -> CrossGridProjectorSchema:
        """Require exactly one of ``file_path`` or ``edges_name``."""
        if (self.file_path is None and self.edges_name is None) or (
            self.file_path is not None and self.edges_name is not None
        ):
            msg = "CrossGridProjectorSchema requires exactly one of 'file_path' or 'edges_name'."
            raise ValueError(msg)
        return self


# Union type for all spatial preprocessors.  Extend this when new
# SpatialPreprocessor subclasses are added; add a discriminator once there
# are multiple members.
SpatialProcessorSchema = Union[CrossGridProjectorSchema]
