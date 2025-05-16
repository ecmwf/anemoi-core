# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from typing import Literal
from typing import Union

from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import NonNegativeInt

from .common_components import GNNModelComponent
from .common_components import TransformerModelComponent


class GNNEncoderSchema(GNNModelComponent):
    target_: Literal["anemoi.models.layers.mapper.GNNForwardMapper"] = Field(..., alias="_target_")
    "GNN encoder object from anemoi.models.layers.mapper."


class GraphTransformerEncoderSchema(TransformerModelComponent):
    target_: Literal["anemoi.models.layers.mapper.GraphTransformerForwardMapper"] = Field(..., alias="_target_")
    "Graph Transfromer Encoder object from anemoi.models.layers.mapper."
    trainable_size: NonNegativeInt = Field(example=8)
    "Size of trainable parameters vector. Default to 8."
    sub_graph_edge_attributes: list[str] = Field(examples=["edge_length", "edge_dirs"])
    "Edge attributes to consider in the encoder features."
    qk_norm: bool = Field(example=False)
    "Normalize the query and key vectors. Default to False."


class TransformerEncoderSchema(TransformerModelComponent):
    target_: Literal["anemoi.models.layers.mapper.TransformerForwardMapper"] = Field(..., alias="_target_")
    "Transformer Encoder object from anemoi.models.layers.mapper."
    window_size: Union[NonNegativeInt, None] = Field(example=512)
    "Attention window size along the longitude axis. Default to 512."
    dropout_p: NonNegativeFloat = Field(example=0.0)
    "Dropout probability used for multi-head self attention, default 0.0"
    attention_implementation: str = Field(example="flash_attention")
    "Attention implementation to use. Default to 'flash_attention'."
    softcap: NonNegativeFloat = Field(example=0.0)
    "Softcap value for attention. Default to 0.0."
    use_alibi_slopes: bool = Field(example=False)
    "Use alibi slopes for attention implementation. Default to False."
