# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from enum import Enum
from typing import Union

from pydantic import Field
from pydantic import NonNegativeInt

from anemoi.training.schemas.utils import BaseModel


class ActivationFunctions(str, Enum):
    GELU = "torch.nn.GELU"
    SiLU = "torch.nn.SiLU"
    ELU = "torch.nn.ELU"
    ReLU = "torch.nn.ReLU"
    Tanh = "torch.nn.Tanh"
    Sigmoid = "torch.nn.Sigmoid"
    Hardshrink = "torch.nn.Hardshrink"
    Hardsigmoid = "torch.nn.Hardsigmoid"
    Hardtanh = "torch.nn.Hardtanh"
    Hardswish = "torch.nn.Hardswish"
    LeakyReLU = "torch.nn.LeakyReLU"
    LogSigmoid = "torch.nn.LogSigmoid"
    PReLU = "torch.nn.PReLU"
    ReLU6 = "torch.nn.ReLU6"
    SELU = "torch.nn.SELU"
    CELU = "torch.nn.CELU"
    Mish = "torch.nn.Mish"
    Softplus = "torch.nn.Softplus"
    Softshrink = "torch.nn.Softshrink"
    Softsign = "torch.nn.Softsign"
    Tanhshrink = "torch.nn.Tanhshrink"
    Threshold = "torch.nn.Threshold"


class GLUActivationFunctions(str, Enum):
    GLU = "anemoi.models.layers.activations.GLU"
    SwiGLU = "anemoi.models.layers.activations.SwiGLU"
    ReGLU = "anemoi.models.layers.activations.ReGLU"
    GeGLU = "anemoi.models.layers.activations.GeGLU"


class ActivationFunctionSchema(BaseModel):
    target_: ActivationFunctions = Field(..., alias="_target_")
    "Activation function class implementation."
    partial_: bool = Field(..., alias="_partial_")
    "Should always be True to avoid using the same activation function object in the different layers."


class GLUActivationFunctionSchema(ActivationFunctionSchema):
    target_: GLUActivationFunctions = Field(..., alias="_target_")
    "Activation function class implementation."
    partial_: bool = Field(..., alias="_partial_")
    "Should always be True to avoid using the same activation function object in the different layers."


class TransformerModelComponent(BaseModel):
    activation: Union[ActivationFunctionSchema, GLUActivationFunctionSchema]
    "Activation function to use for the transformer model component. Default to GELU."
    convert_: str = Field("all", alias="_convert_")
    "Target's parameters to convert to primitive containers. Other parameters will use OmegaConf. Default to all."
    cpu_offload: bool = Field(example=False)
    "Offload to CPU. Default to False."
    num_chunks: NonNegativeInt = Field(example=1)
    "Number of chunks to divide the layer into. Default to 1."
    mlp_hidden_ratio: NonNegativeInt = Field(example=4)
    "Ratio of mlp hidden dimension to embedding dimension. Default to 4."
    num_heads: NonNegativeInt = Field(example=16)
    "Number of attention heads. Default to 16."


class GNNModelComponent(BaseModel):
    activation: Union[ActivationFunctionSchema, GLUActivationFunctionSchema]
    "Activation function to use for the GNN model component. Default to GELU."
    convert_: str = Field("all", alias="_convert_")
    "Target's parameters to convert to primitive containers. Other parameters will use OmegaConf. Default to all."
    trainable_size: NonNegativeInt = Field(example=8)
    "Size of trainable parameters vector. Default to 8."
    num_chunks: NonNegativeInt = Field(example=1)
    "Number of chunks to divide the layer into. Default to 1."
    cpu_offload: bool = Field(example=False)
    "Offload to CPU. Default to False."
    sub_graph_edge_attributes: list[str] = Field(default_factory=list)
    "Edge attributes to consider in the model component features."
    mlp_extra_layers: NonNegativeInt = Field(example=0)
    "The number of extra hidden layers in MLP. Default to 0."
