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
    qk_norm: bool = Field(example=False)
    "Normalize the query and key vectors. Default to False."
    initialise_data_extractor_zero: bool = Field(example=False)
    "Initialise the data extractor with zeros. Default to False."


class TransformerDecoderSchema(TransformerModelComponent):
    target_: Literal["anemoi.models.layers.mapper.TransformerBackwardMapper"] = Field(..., alias="_target_")
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
    use_rotary_embeddings: bool = Field(example=False)
    "Use rotary embeddings for attention implementation. Default to False."
