# (C) Copyright 2024-2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import logging
import sys
from typing import Any

from omegaconf import DictConfig
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field
from pydantic_core import ValidationError

from anemoi.utils.schemas import BaseModel
from anemoi.utils.schemas.errors import CUSTOM_MESSAGES
from anemoi.utils.schemas.errors import convert_errors

from .edge_attributes_schemas import EdgeAttributeSchema  # noqa: TC001
from .edge_schemas import EdgeBuilderSchemas  # noqa: TC001
from .node_attributes_schemas import NodeAttributeSchemas  # noqa: TC001
from .node_schemas import NodeBuilderSchemas  # noqa: TC001
from .post_processors import ProcessorSchemas  # noqa: TC001

LOGGER = logging.getLogger(__name__)


class NodeSchema(BaseModel):
    node_builder: NodeBuilderSchemas
    "Node builder schema."
    attributes: dict[str, NodeAttributeSchemas] = Field(default_factory=dict)
    "Dictionary of attributes with names as keys and anemoi.graphs.nodes.attributes objects as values."


class EdgeSchema(BaseModel):
    source_name: str = Field(examples=["data", "hidden"])
    "Source of the edges."
    target_name: str = Field(examples=["data", "hidden"])
    "Target of the edges."
    edge_builders: list[EdgeBuilderSchemas]
    "Edge builder schema."
    attributes: dict[str, EdgeAttributeSchema] = Field(default_factory=dict)
    "Dictionary of attributes with names as keys and anemoi.graphs.edges.attributes objects as values."


class BaseGraphSchema(PydanticBaseModel):
    nodes: dict[str, NodeSchema] = Field(default_factory=dict)
    "Nodes schema for all types of nodes (ex. data, hidden)."
    edges: list[EdgeSchema] = Field(default_factory=list)
    "List of edges schema."
    post_processors: list[ProcessorSchemas] = Field(default_factory=list)

    def model_dump(self, by_alias: bool = False) -> DictConfig:
        dumped_model = super().model_dump(by_alias=by_alias)
        return DictConfig(dumped_model)


class UnvalidatedGraphSchema(PydanticBaseModel):
    """Unvalidated graph schema for the training configuration."""

    nodes: Any
    "Nodes schema for all types of nodes (ex. data, hidden)."
    edges: Any
    "List of edges schema."

    def model_dump(self, by_alias: bool = False) -> DictConfig:
        dumped_model = super().model_dump(by_alias=by_alias)
        return DictConfig(dumped_model)


def validate_graph_schema(config: DictConfig) -> BaseGraphSchema:
    try:
        config = BaseGraphSchema(**config)
    except ValidationError as e:
        errors = convert_errors(e, CUSTOM_MESSAGES)
        LOGGER.error(errors)  # noqa: TRY400
        sys.exit(0)
    else:
        return config
