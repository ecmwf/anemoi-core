# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from typing import Any
from typing import Literal

from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import NonNegativeInt
from pydantic import ValidationError
from pydantic import model_validator

from .common_components import GNNModelComponent
from .common_components import TransformerModelComponent


class GNNProcessorSchema(GNNModelComponent):
    target_: Literal["anemoi.models.layers.processor.GNNProcessor"] = Field(..., alias="_target_")
    "GNN Processor object from anemoi.models.layers.processor."
    num_layers: NonNegativeInt = Field(example=16)
    "Number of layers of GNN processor. Default to 16."
    num_chunks: NonNegativeInt = Field(example=2)
    "Number of chunks to divide the layer into. Default to 2."


class GraphTransformerProcessorSchema(TransformerModelComponent):
    target_: Literal["anemoi.models.layers.processor.GraphTransformerProcessor"] = Field(..., alias="_target_")
    "Graph transformer processor object from anemoi.models.layers.processor."
    trainable_size: NonNegativeInt = Field(example=8)
    "Size of trainable parameters vector. Default to 8."
    sub_graph_edge_attributes: list[str] = Field(example=["edge_length", "edge_dir"])
    "Edge attributes to consider in the processor features. Default [edge_length, endge_dirs]."
    num_layers: NonNegativeInt = Field(example=16)
    "Number of layers of Graph Transformer processor. Default to 16."
    num_chunks: NonNegativeInt = Field(example=2)
    "Number of chunks to divide the layer into. Default to 2."
    qk_norm: bool = Field(example=False)
    "Normalize the query and key vectors. Default to False."


class TransformerProcessorSchema(TransformerModelComponent):

    target_: Literal["anemoi.models.layers.processor.TransformerProcessor"] = Field(..., alias="_target_")
    "Transformer processor object from anemoi.models.layers.processor."
    num_layers: NonNegativeInt = Field(example=16)
    "Number of layers of Transformer processor. Default to 16."
    num_chunks: NonNegativeInt = Field(example=2)
    "Number of chunks to divide the layer into. Default to 2."
    window_size: NonNegativeInt = Field(example=512)
    "Attention window size along the longitude axis. Default to 512."
    dropout_p: NonNegativeFloat = Field(example=0.0)
    "Dropout probability used for multi-head self attention, default 0.0"
    attention_implementation: str = Field(example="flash_attention")
    "Attention implementation to use. Default to 'flash_attention'."
    qk_norm: bool = Field(example=False)
    "Normalize the query and key vectors. Default to False."
    softcap: NonNegativeFloat = Field(example=0.0)
    "Softcap value for attention. Default to 0.0."
    use_alibi_slopes: bool = Field(example=False)
    "Use alibi slopes for attention implementation. Default to False."

    @model_validator(mode="after")
    def check_valid_extras(self) -> Any:
        # Check for valid extra fields related to MultiHeadSelfAttention and MultiHeadCrossAttention
        # This is a check to allow backwards compatibilty of the configs, as the extra fields are not required.
        allowed_extras = {"use_rotary_embeddings": bool}
        for extra_field in self.__pydantic_extra__:
            if extra_field not in allowed_extras:
                msg = f"Extra field {extra_field} not allowed for TransformerProcessorSchema."
                raise ValidationError(msg)
            if isinstance(extra_field, allowed_extras[extra_field]):
                msg = f"Extra field {extra_field} should be of type {allowed_extras[extra_field]}."
                raise ValidationError(msg)

        return self
