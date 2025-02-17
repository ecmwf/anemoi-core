# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from typing import Literal

from pydantic import Field
from pydantic import NonNegativeInt

from .common_components import GNNModelComponent
from .common_components import TransformerModelComponent


class GNNDecoderSchema(GNNModelComponent):
    target_: Literal["anemoi.models.layers.mapper.GNNBackwardMapper"] = Field(..., alias="_target_")
    "GNN decoder object from anemoi.models.layers.mapper."


class GraphTransformerDecoderSchema(TransformerModelComponent):
    target_: Literal["anemoi.models.layers.mapper.GraphTransformerBackwardMapper"] = Field(..., alias="_target_")
    "Graph Transformer Decoder object from anemoi.models.layers.mapper."
    trainable_size: NonNegativeInt = Field(example=8)
    "Size of trainable parameters vector. Default to 8."
    sub_graph_edge_attributes: list[str] = Field(example=["edge_length", "edge_dirs"])
    "Edge attributes to consider in the decoder features. Default to [edge_length, edge_dirs]"
