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

from pydantic import BaseModel
from pydantic import Field

from .edge_attributes_schemas import EdgeAttributeSchema  # noqa: TC001
from .edge_schemas import EdgeBuilderSchemas  # noqa: TC001
from .node_attributes_schemas import NodeAttributeSchemas  # noqa: TC001
from .node_schemas import NodeBuilderSchemas  # noqa: TC001
from .post_processors import ProcessorSchema  # noqa: TC001

LOGGER = logging.getLogger(__name__)


class NodeSchema(BaseModel):
    node_builder: NodeBuilderSchemas
    "Node builder schema."
    attributes: dict[str, NodeAttributeSchemas] | None = None
    "Dictionary of attributes with names as keys and anemoi.graphs.nodes.attributes objects as values."


class EdgeSchema(BaseModel):
    source_name: str = Field(examples=["data", "hidden"])
    "Source of the edges."
    target_name: str = Field(examples=["data", "hidden"])
    "Target of the edges."
    edge_builders: list[EdgeBuilderSchemas]
    "Edge builder schema."
    attributes: dict[str, EdgeAttributeSchema]
    "Dictionary of attributes with names as keys and anemoi.graphs.edges.attributes objects as values."


class BaseGraphSchema(BaseModel):
    nodes: dict[str, NodeSchema]
    "Nodes schema for all types of nodes (ex. data, hidden)."
    edges: list[EdgeSchema]
    "List of edges schema."
    overwrite: bool = Field(default=True)
    "whether to overwrite existing graph file. Default to True."
    post_processors: list[ProcessorSchema]
    data: str = Field(default="data")
    "Key name for the data nodes. Default to 'data'."
    hidden: str = Field(default="hidden")
    "Key name for the hidden nodes. Default to 'hidden'."
    # TODO(Helen): Needs to be adjusted for more complex graph setups
