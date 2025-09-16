# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import logging
from typing import Annotated
from typing import Iterable
from typing import Literal

from pydantic import Field

from anemoi.utils.schemas import BaseModel

LOGGER = logging.getLogger(__name__)


class RemoveUnconnectedNodesSchema(BaseModel):
    target_: Literal["anemoi.graphs.processors.RemoveUnconnectedNodes"] = Field(..., alias="_target_")
    "Post processor to remove unconnected nodes."
    nodes_name: str | Iterable[str]
    "Nodes from which to remove the unconnected nodes."
    ignore: str = Field(example=None)
    "Attribute name of nodes to be ignored."
    save_mask_indices_to_attr: str = Field(example=None)
    "New attribute name to store the mask indices."


class SubsetNodesInAreaSchema(BaseModel):
    target_: Literal["anemoi.graphs.processors.SubsetNodesInArea"] = Field(..., alias="_target_")
    "Post processor to remove unconnected nodes."
    nodes_name: str | Iterable[str]
    "Nodes from which to remove the unconnected nodes."
    area: tuple[float, float, float, float] = Field(default=(40, 10, 30, 20))
    "Area of interest to crop the nodes, (north, west, south, east)."
    save_mask_indices_to_attr: str = Field(example=None)
    "New attribute name to store the mask indices."


class RestrictEdgeLengthSchema(BaseModel):
    target_: Literal["anemoi.graphs.processors.RestrictEdgeLength"] = Field(..., alias="_target_")
    "Post processor to edges longer than a threshold."
    source_name: str
    "Source nodes of edges to be post-processed."
    target_name: str
    "Target nodes of edges to be post-processed."
    max_length_km: float
    "Treshold length (in km), edges longer than this length will be removed"
    source_mask_attr_name: str | None = Field(default=None, example=None)
    "Boolean mask attribute on sources nodes. Only edges whose source is True under this mask will be post-processed. Default to None"
    target_mask_attr_name: str | None = Field(default=None, example=None)
    "Boolean mask attribute on target nodes. Only edges whose target is True under this mask will be post-processed. Default to None"


class SortEdgeIndexSchema(BaseModel):
    target_: Literal[
        "anemoi.graphs.processors.SortEdgeIndexBySourceNodes",
        "anemoi.graphs.processors.SortEdgeIndexByTargetNodes",
    ] = Field(..., alias="_target_")
    "Post processor to sort edge indices based on either source or target nodes."
    descending: bool = Field(default=True, example=True)
    "Flag to sort edge indices in descending order."


ProcessorSchemas = Annotated[
    RemoveUnconnectedNodesSchema | SubsetNodesInAreaSchema | RestrictEdgeLengthSchema | SortEdgeIndexSchema,
    Field(discriminator="target_"),
]
