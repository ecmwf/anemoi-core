# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from pydantic import Field

from anemoi.graphs.schemas.base_graph import BaseGraphSchema
from anemoi.utils.schemas import BaseModel

if TYPE_CHECKING:
    from pathlib import Path


class GraphAssetSchema(BaseModel):
    """Schema for an auxiliary graph asset."""

    graph_config: BaseGraphSchema
    file_path: Path | None = Field(
        default=None,
        description="Optional path to load/save the asset graph (.pt). If set, dataset suffixing applies.",
    )
    overwrite: bool = Field(default=False, description="Whether to overwrite existing asset graph file.")


class TrainingGraphSchema(BaseGraphSchema):
    """Training graph schema with assets and providers."""

    assets: dict[str, GraphAssetSchema] | None = Field(default=None)
    providers: dict[str, Any] | None = Field(default=None)
