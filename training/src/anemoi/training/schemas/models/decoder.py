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
from typing import Union

from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import NonNegativeInt
from pydantic import ValidationError
from pydantic import model_validator

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

    @model_validator(mode="after")
    def check_valid_extras(self) -> Any:
        # Check for valid extra fields
        # This is a check to allow backwards compatibilty of the configs, as the extra fields are not required.
        # Please extend as needed.
        allowed_extras = {}  # list of allowed extra fields
        for extra_field in self.__pydantic_extra__:
            if extra_field not in allowed_extras:
                msg = f"Extra field {extra_field} not allowed for TransformerProcessorSchema."
                raise ValidationError(msg)
            if isinstance(extra_field, allowed_extras[extra_field]):
                msg = f"Extra field {extra_field} should be of type {allowed_extras[extra_field]}."
                raise ValidationError(msg)

        return self


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

    @model_validator(mode="after")
    def check_valid_extras(self) -> Any:
        # Check for valid extra fields related to MultiHeadSelfAttention and MultiHeadCrossAttention
        # This is a check to allow backwards compatibilty of the configs, as the extra fields are not required.
        allowed_extras = {"use_qk_norm": bool, "use_rotary_embeddings": bool}
        for extra_field in self.__pydantic_extra__:
            if extra_field not in allowed_extras:
                msg = f"Extra field {extra_field} not allowed for TransformerProcessorSchema."
                raise ValidationError(msg)
            if isinstance(extra_field, allowed_extras[extra_field]):
                msg = f"Extra field {extra_field} should be of type {allowed_extras[extra_field]}."
                raise ValidationError(msg)

        return self
