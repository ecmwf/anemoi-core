# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from enum import Enum

from pydantic import Field
from pydantic import NonNegativeInt

from .utils import BaseModel


class ActivationFunctons(str, Enum):
    GELU = "GELU"
    SiLU = "SiLU"
    ELU = "ELU"
    ReLU = "ReLU"
    Tanh = "Tanh"
    Sigmoid = "Sigmoid"
    Hardshrink = "Hardshrink"
    Hardsigmoid = "Hardsigmoid"
    Hardtanh = "Hardtanh"
    Hardswish = "Hardswish"
    LeakyReLU = "LeakyReLU"
    LogSigmoid = "LogSigmoid"
    PReLU = "PReLU"
    ReLU6 = "ReLU6"
    SELU = "SELU"
    CELU = "CELU"
    Mish = "Mish"
    Softplus = "Softplus"
    Softshrink = "Softshrink"
    Softsign = "Softsign"
    Tanhshrink = "Tanhshrink"
    Threshold = "Threshold"
    GLU = "GLU"


class TransformerModelComponent(BaseModel):
    activation: ActivationFunctons = Field(example="GELU")
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
    activation: ActivationFunctons = Field(example="GELU")
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
