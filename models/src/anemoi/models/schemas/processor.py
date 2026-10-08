# (C) Copyright 2024- Anemoi contributors.
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
from pydantic import NonNegativeFloat
from pydantic import NonNegativeInt
from pydantic import PositiveFloat
from pydantic import PositiveInt
from pydantic import model_validator

from .common_components import GNNModelComponent
from .common_components import PointWiseModelComponent
from .common_components import TransformerModelComponent


class NoOpProcessorSchema(PydanticBaseModel):
    target_: Literal["anemoi.models.layers.processor.NoOpProcessor"] = Field(..., alias="_target_")
    "No-op processor, used for ablations."


class GNNProcessorSchema(GNNModelComponent):
    target_: Literal["anemoi.models.layers.processor.GNNProcessor"] = Field(..., alias="_target_")
    "GNN Processor object from anemoi.models.layers.processor."
    num_channels: NonNegativeInt = Field(example=512)
    "Number of channels in the GNN processor. Default to 512."
    num_layers: NonNegativeInt = Field(example=16)
    "Number of layers of GNN processor. Default to 16."
    num_chunks: NonNegativeInt = Field(example=2)
    "Number of chunks to divide the layer into. Default to 2."


class GraphTransformerProcessorSchema(TransformerModelComponent):
    target_: Literal["anemoi.models.layers.processor.GraphTransformerProcessor"] = Field(..., alias="_target_")
    "Graph transformer processor object from anemoi.models.layers.processor."
    num_channels: NonNegativeInt = Field(example=512)
    "Number of channels in the Graph Transformer processor. Default to 512."
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

    @model_validator(mode="after")
    def check_valid_extras(self) -> Any:
        # This is a check to allow backwards compatibilty of the configs, as the extra fields are not required.
        allowed_extras = {
            "shard_strategy": str,
            "graph_attention_backend": str,
            "edge_pre_mlp": bool,
            "gradient_checkpointing": bool,
        }
        extras = getattr(self, "__pydantic_extra__", {}) or {}
        for extra_field, value in extras.items():
            if extra_field not in allowed_extras:
                msg = f"Extra field '{extra_field}' is not allowed. Allowed fields are: {list(allowed_extras.keys())}."
                raise ValueError(msg)
            if not isinstance(value, allowed_extras[extra_field]):
                msg = f"Extra field '{extra_field}' must be of type {allowed_extras[extra_field].__name__}."
                raise TypeError(msg)

        return self


class TransformerProcessorSchema(TransformerModelComponent):
    target_: Literal["anemoi.models.layers.processor.TransformerProcessor"] = Field(..., alias="_target_")
    "Transformer processor object from anemoi.models.layers.processor."
    num_channels: NonNegativeInt = Field(example=512)
    "Number of channels in the Transformer processor. Default to 512."
    num_layers: NonNegativeInt = Field(example=16)
    "Number of layers of Transformer processor. Default to 16."
    num_chunks: NonNegativeInt = Field(example=2)
    "Number of chunks to divide the layer into. Default to 2."
    window_size: Union[NonNegativeInt, None] = Field(example=512)
    "Attention window size along the longitude axis. Default to 512."
    dropout_p: NonNegativeFloat = Field(example=0.0)
    "Dropout probability used for multi-head self attention, default 0.0"
    attention_implementation: str = Field(example="triton_attention")
    "Attention implementation to use. Default to 'triton_attention'."
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
        allowed_extras = {"use_rotary_embeddings": bool, "gradient_checkpointing": bool}
        extras = getattr(self, "__pydantic_extra__", {}) or {}
        for extra_field, value in extras.items():
            if extra_field not in allowed_extras:
                msg = f"Extra field '{extra_field}' is not allowed. Allowed fields are: {list(allowed_extras.keys())}."
                raise ValueError(msg)
            if not isinstance(value, allowed_extras[extra_field]):
                msg = f"Extra field '{extra_field}' must be of type {allowed_extras[extra_field].__name__}."
                raise TypeError(msg)

        return self


class ADRProcessorSchema(PointWiseModelComponent):
    target_: Literal["anemoi.models.layers.processor.ADRProcessor"] = Field(..., alias="_target_")
    "Advection-diffusion-reaction processor on a regular latitude-longitude hidden grid (PARADIS). The grid size is read from the hidden graph."
    num_channels: NonNegativeInt = Field(example=1024)
    "Number of channels in the processor."
    num_layers: NonNegativeInt = Field(example=16)
    "Number of advection-diffusion-reaction layers."
    timestep: str = Field(example="6h")
    "Model time step. Sets how far the learned velocities move the state."
    advection_channels: PositiveInt = Field(default=768)
    "Number of channels moved by the advection step."
    num_heads: Union[PositiveInt, None] = Field(default=None)
    "Number of velocity fields. Default: one per moved channel."
    velocity_hidden_dim: PositiveInt = Field(default=384)
    "Hidden dimension of the velocity network."
    reaction_hidden_dim: PositiveInt = Field(default=896)
    "Hidden dimension of the reaction MLP."
    reaction_num_layers: PositiveInt = Field(default=4)
    "Number of linear layers in the reaction MLP, at least 2."
    kernel_size: PositiveInt = Field(default=5)
    "Size of the square stencils of the spatial mixers, must be odd."
    interpolation: Literal["bicubic", "bilinear"] = Field(default="bicubic")
    "Interpolation used to read values at departure points."
    bias_rank: PositiveInt = Field(default=128)
    "Number of latitude-longitude profile pairs in each learned bias field."
    bias_base_maps: PositiveInt = Field(default=8)
    "Number of base maps each learned bias field is mixed from."
    cartesian_displacement: bool = Field(default=False)
    "Predict velocities as 3D vectors instead of local east and north components."


class FlowersProcessorSchema(PointWiseModelComponent):
    target_: Literal["anemoi.models.layers.processor.FlowersProcessor"] = Field(..., alias="_target_")
    "Processor of FLOWERS warp blocks on a regular latitude-longitude hidden grid. The grid size is read from the hidden graph."
    num_channels: NonNegativeInt = Field(example=512)
    "Number of channels in the processor."
    num_layers: NonNegativeInt = Field(example=8)
    "Number of warp blocks."
    num_heads: Union[PositiveInt, None] = Field(default=None)
    "Number of displacement fields per block. Default: 4 channels per head."
    mlp_hidden_ratio: PositiveFloat = Field(default=4.0)
    "Ratio of the MLP hidden dimension to num_channels, used by the pre_norm block."
    mlp_implementation: Literal["mlp", "glu", "swiglu", "geglu", "reglu"] = Field(default="mlp")
    "Implementation of the MLP in the pre_norm block."
    block_style: Literal["pre_norm", "flowers"] = Field(default="pre_norm")
    "Residual warp and MLP after layer norms, or the original FLOWERS block."
    interpolation: Literal["bicubic", "bilinear"] = Field(default="bilinear")
    "Interpolation used to read values at departure points."
    cartesian_displacement: bool = Field(default=False)
    "Predict displacements as 3D vectors instead of local east and north angles."


class PointWiseMLPProcessorSchema(PointWiseModelComponent):
    target_: Literal["anemoi.models.layers.processor.PointWiseMLPProcessor"] = Field(..., alias="_target_")
    "PointWise MLP processor object from anemoi.models.layers.processor."
    num_channels: NonNegativeInt = Field(example=512)
    "Number of channels in the PointWise MLP processor. Default to 512."
    num_layers: NonNegativeInt = Field(example=16)
    "Number of layers of the PointWise MLP processor."
    mlp_hidden_ratio: NonNegativeInt = Field(example=4)
    "Ratio of the hidden dimension to the processor channel dimension."
    dropout_p: NonNegativeFloat = Field(default=0.0, example=0.0)
    "Dropout probability, default 0.0"
