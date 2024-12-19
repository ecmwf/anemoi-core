# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from typing import Literal

from pydantic import Field

from .common_components import GNNModelComponent
from .common_components import TransformerModelComponent


class GNNEncoder(GNNModelComponent):
    target_: Literal["anemoi.models.layers.mapper.GNNForwardMapper"] = Field(..., alias="_target_")
    "GNN encoder object from anemoi.models.layers.mapper."


class GraphTransformerEncoder(TransformerModelComponent):
    target_: Literal["anemoi.models.layers.mapper.GraphTransformerForwardMapper"] = Field(..., alias="_target_")
    "Graph Transfromer Encoder object from anemoi.models.layers.mapper."
    sub_graph_edge_attributes: list[str] = Field(default=["edge_length", "edge_dirs"])
    "Edge attributes to consider in the encoder features."
