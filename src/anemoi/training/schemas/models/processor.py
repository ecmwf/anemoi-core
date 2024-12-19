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
from pydantic import NonNegativeFloat
from pydantic import NonNegativeInt

from .common_components import GNNModelComponent
from .common_components import TransformerModelComponent


class GNNProcessor(GNNModelComponent):
    target_: Literal["anemoi.models.layers.processor.GNNProcessor"] = Field(..., alias="_target_")
    "GNN Processor object from anemoi.models.layers.processor."
    num_layers: NonNegativeInt = Field(default=16)
    "Number of layers of GNN processor. Default to 16."
    num_chunks: NonNegativeInt = Field(default=2)
    "Number of chunks to divide the layer into. Default to 2."


class GraphTransformerProcessor(TransformerModelComponent):
    target_: Literal["anemoi.models.layers.processor.GraphTransformerProcessor"] = Field(..., alias="_target_")
    "Graph transformer processor object from anemoi.models.layers.processor."
    sub_graph_edge_attributes: list[str] = Field(default=["edge_length", "edge_dir"])
    "Edge attributes to consider in the processor features. Default [edge_length, endge_dirs]."
    num_layers: NonNegativeInt = Field(default=16)
    "Number of layers of Graph Transformer processor. Default to 16."
    num_chunks: NonNegativeInt = Field(default=2)
    "Number of chunks to divide the layer into. Default to 2."
    dropout_p: NonNegativeFloat = Field(default=0.0)
    "Dropout probability used for multi-head self attention, default 0.0"


class TransformerProcessor(TransformerModelComponent):
    target_: Literal["anemoi.models.layers.processor.TransformerProcessor"] = Field(..., alias="_target_")
    "Transformer processor object from anemoi.models.layers.processor."
    num_layers: NonNegativeInt = Field(default=16)
    "Number of layers of Transformer processor. Default to 16."
    num_chunks: NonNegativeInt = Field(default=2)
    "Number of chunks to divide the layer into. Default to 2."
    window_size: NonNegativeInt = Field(default=512)
    "Attention window size along the longitude axis. Default to 512."
    dropout_p: NonNegativeFloat = Field(default=0.0)
    "Dropout probability used for multi-head self attention, default 0.0"
