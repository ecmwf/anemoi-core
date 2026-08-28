# (C) Copyright 2026 Anemoi contributors.
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
from pydantic import PositiveInt

from .common_components import TransformerModelComponent


class GraphTransformerPreMixerSchema(TransformerModelComponent):
    target_: Literal["anemoi.models.layers.premixer.GraphTransformerPreMixer"] = Field(..., alias="_target_")
    "Graph transformer pre-mixer object from anemoi.models.layers.premixer."
    num_channels: PositiveInt = Field(example=256)
    "Internal mixing width. Independent of model.num_channels."
    num_layers: PositiveInt = Field(example=4)
    "Number of point-to-point mixing layers."
    trainable_size: NonNegativeInt = Field(default=0)
    "Size of trainable parameters vector on the data -> data edges. Default 0: that graph has "
    "one edge per data node per neighbour, so a non-zero value adds millions of parameters."
    sub_graph_edge_attributes: list[str] = Field(default_factory=list, examples=["edge_length", "edge_dirs"])
    "Edge attributes to consider in the pre-mixer features."
    qk_norm: bool = Field(default=False)
    "Normalize the query and key vectors. Default to False."
    initialise_out_zero: bool = Field(default=True)
    "Zero-initialise the output projection so the module starts as an exact identity. "
    "Keep True when forking a checkpoint trained without a pre-mixer."
    graph_attention_backend: str = Field(default="triton")
    "Backend for the graph attention, 'triton' or 'pyg'."
    edge_pre_mlp: bool = Field(default=False)
    "Allow for edge feature mixing. Default to False."
