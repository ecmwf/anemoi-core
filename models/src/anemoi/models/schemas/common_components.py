# (C) Copyright 2024-2026 Anemoi contributors.
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

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveFloat
from pydantic import PositiveInt

from anemoi.utils.schemas import BaseModel


class RotaryEmbeddingsSchema(BaseModel):
    max_frequency: float = Field(ge=1.0, example=10000.0)
    "Highest frequency in radians per Earth radius; it repeats every 40,000 km / max_frequency. The lowest is 1."
    backend: Literal["torch"] = Field(default="torch")
    "Rotate with PyTorch ('torch', any device). Default to 'torch'."


def check_neighbourhood_attention(component: Any) -> Any:
    """Check that the neighbourhood settings and the attention implementation of a component agree."""
    if component.attention_implementation != "neighbourhood":
        if component.neighbourhood is not None:
            raise ValueError("'neighbourhood' is only used with attention_implementation 'neighbourhood'.")
        return component
    if component.neighbourhood is None:
        raise ValueError("attention_implementation 'neighbourhood' needs a 'neighbourhood' section.")
    if component.window_size is not None:
        raise ValueError("Neighbourhood attention sets its own mask; window_size must be null.")
    return component


class TransformerModelComponent(PydanticBaseModel):
    class Config:
        """Pydantic BaseModel configuration."""

        use_attribute_docstrings = True
        use_enum_values = True
        validate_assignment = True
        validate_default = True
        extra = "allow"  # Beware this allows extra fields in the config, typos are less likely to be spotted

    convert_: str = Field("all", alias="_convert_")
    "Target's parameters to convert to primitive containers. Other parameters will use OmegaConf. Default to all."
    cpu_offload: bool = Field(example=False)
    "Offload to CPU. Default to False."
    gradient_checkpointing: bool = Field(default=True)
    "Enable gradient checkpointing to reduce memory usage. Default to True."
    num_chunks: NonNegativeInt = Field(example=1)
    "Number of chunks to divide the layer into. Default to 1."
    mlp_hidden_ratio: PositiveFloat = Field(example=4)
    "Ratio of MLP hidden dimension to embedding dimension. Use 4 for `mlp`, and ~2.67 for gated variants (`glu`, `swiglu`, `geglu`, `reglu`) to keep model size similar."
    mlp_implementation: Literal["mlp", "glu", "swiglu", "geglu", "reglu"] = Field(default="mlp", example="mlp")
    "Implementation of feed-forward blocks (`mlp`, `glu`, `swiglu`, `geglu`, `reglu`). Default to `mlp`."
    num_heads: NonNegativeInt = Field(example=16)
    "Number of attention heads. Default to 16."
    attn_channels: Union[PositiveInt, None] = Field(default=None)
    "Internal attention width used for q/k/v projections. Default to None, which keeps the embedding dimension."
    layer_kernels: Union[dict[str, dict], None] = Field(default_factory=dict)
    "Settings related to custom kernels for encoder processor and decoder blocks"


class GNNModelComponent(BaseModel):
    convert_: str = Field("all", alias="_convert_")
    "Target's parameters to convert to primitive containers. Other parameters will use OmegaConf. Default to all."
    trainable_size: NonNegativeInt = Field(example=8)
    "Size of trainable parameters vector. Default to 8."
    num_chunks: NonNegativeInt = Field(example=1)
    "Number of chunks to divide the layer into. Default to 1."
    cpu_offload: bool = Field(example=False)
    "Offload to CPU. Default to False."
    gradient_checkpointing: bool = Field(default=True)
    "Enable gradient checkpointing to reduce memory usage. Default to True."
    sub_graph_edge_attributes: list[str] = Field(default_factory=list)
    "Edge attributes to consider in the model component features."
    mlp_extra_layers: NonNegativeInt = Field(example=0)
    "The number of extra hidden layers in MLP. Default to 0."
    mlp_hidden_ratio: PositiveFloat = Field(default=1.0, example=1.0)
    "Ratio of MLP hidden dimension to channel width. Use 1.0 for no expansion, ~2.67 for gated variants to match transformer parameter counts."
    mlp_implementation: Literal["mlp", "glu", "swiglu", "geglu", "reglu"] = Field(default="mlp", example="mlp")
    "Implementation of feed-forward blocks (`mlp`, `glu`, `swiglu`, `geglu`, `reglu`). Default to `mlp`."
    layer_kernels: Union[dict[str, dict], None] = Field(default_factory=dict)
    "Settings related to custom kernels for encoder processor and decoder blocks"


class PointWiseModelComponent(BaseModel):
    convert_: str = Field("all", alias="_convert_")
    "Target's parameters to convert to primitive containers. Other parameters will use OmegaConf. Default to all."
    num_chunks: NonNegativeInt = Field(example=1)
    "Number of chunks to divide the layer into. Default to 1."
    cpu_offload: bool = Field(example=False)
    "Offload to CPU. Default to False."
    gradient_checkpointing: bool = Field(default=True)
    "Enable gradient checkpointing to reduce memory usage. Default to True."
    layer_kernels: Union[dict[str, dict], None] = Field(default_factory=dict)
    "Settings related to custom kernels for encoder processor and decoder blocks"


class PointWiseMapperComponent(BaseModel):
    convert_: str = Field("all", alias="_convert_")
    "Target's parameters to convert to primitive containers. Other parameters will use OmegaConf. Default to all."
    cpu_offload: bool = Field(example=False)
    "Offload to CPU. Default to False."
    gradient_checkpointing: bool = Field(default=True)
    "Enable gradient checkpointing to reduce memory usage. Default to True."
    layer_kernels: Union[dict[str, dict], None] = Field(default_factory=dict)
    "Settings related to custom kernels for encoder and decoder blocks"
