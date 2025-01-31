# (C) Copyright 2024-2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import Field


class EdgeAttributeSchema(BaseModel):
    target_: Literal["anemoi.graphs.edges.attributes.EdgeLength", "anemoi.graphs.edges.attributes.EdgeDirection"] = (
        Field("anemoi.graphs.edges.attributes.EdgeLength", alias="_target_")
    )
    "Edge attributes object from anemoi.graphs.edges."
    norm: Literal["unit-max", "l1", "l2", "unit-sum", "unit-std"] = Field(example="unit-std")
    "Normalisation method applied to the edge attribute."
