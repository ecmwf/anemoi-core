# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

import logging
from enum import Enum
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import NonNegativeInt

from .decoder import GNNDecoderSchema
from .decoder import GraphTransformerDecoderSchema
from .encoder import GNNEncoderSchema
from .encoder import GraphTransformerEncoderSchema
from .processor import GNNProcessorSchema
from .processor import GraphTransformerProcessorSchema
from .processor import TransformerProcessorSchema

LOGGER = logging.getLogger(__name__)


class DefinedModels(str, Enum):
    ANEMOI_MODEL_ENC_PROC_DEC = "anemoi.models.models.encoder_processor_decoder.AnemoiModelEncProcDec"
    ANEMOI_MODEL_ENC_HIERPROC_DEC = "anemoi.models.models.hierarchical.AnemoiModelEncProcDecHierarchical"


class Model(BaseModel):
    target_: DefinedModels = Field(..., alias="_target_")
    "Model object defined in anemoi.models.model."
    convert_: str = Field("all", alias="_convert_")
    "The target's parameters to convert to primitive containers. Other parameters will use OmegaConf. Default to all."


class TrainableParameters(BaseModel):
    data: NonNegativeInt = Field(example=8)
    "Size of the learnable data node tensor. Default to 8."
    hidden: NonNegativeInt = Field(example=8)
    "Size of the learnable hidden node tensor. Default to 8."


class ReluBoundingSchema(BaseModel):
    target_: Literal["anemoi.models.layers.bounding.ReluBounding"] = Field(..., alias="_target_")
    "Relu bounding object defined in anemoi.models.layers.bounding."
    variables: list[str]
    "List of variables to bound using the Relu method."


class FractionBoundingSchema(BaseModel):
    target_: Literal["anemoi.models.layers.bounding.FractionBounding"] = Field(..., alias="_target_")
    "Fraction bounding object defined in anemoi.models.layers.bounding."
    variables: list[str]
    "List of variables to bound using the hard tanh fraction method."
    min_val: float
    "The minimum value for the HardTanh activation. Correspond to the minimum fraction of the total_var."
    max_val: float
    "The maximum value for the HardTanh activation. Correspond to the maximum fraction of the total_var."
    total_var: str
    "Variable from which the secondary variables are derived. \
    For example, convective precipitation should be a fraction of total precipitation."


class HardtanhBoundingSchema(BaseModel):
    target_: Literal["anemoi.models.layers.bounding.HardtanhBounding"] = Field(..., alias="_target_")
    "Hard tanh bounding method function from anemoi.models.layers.bounding."
    variables: list[str]
    "List of variables to bound using the hard tanh method."
    min_val: float
    "The minimum value for the HardTanh activation."
    max_val: float
    "The maximum value for the HardTanh activation."


class BaseModelConfig(BaseModel):
    num_channels: NonNegativeInt = Field(example=512)
    "Feature tensor size in the hidden space."
    model: Model = Field(default_factory=Model)
    "Model schema."
    trainable_parameters: TrainableParameters = Field(default_factory=TrainableParameters)
    "Learnable node and edge parameters."
    bounding: list[ReluBoundingSchema | HardtanhBoundingSchema | FractionBoundingSchema]
    "List of bounding configuration applied in order to the specified variables."
    output_mask: str | None = Field(example=None)  # !TODO CHECK!
    "Output mask, it must be a node attribute of the output nodes"


class GNNSchema(BaseModelConfig):
    processor: GNNProcessorSchema = Field(default_factory=GNNProcessorSchema)
    "GNN processor schema."
    encoder: GNNEncoderSchema = Field(default_factory=GNNEncoderSchema)
    "GNN encoder schema."
    decoder: GNNDecoderSchema = Field(default_factory=GNNDecoderSchema)
    "GNN decoder schema."


class GraphTransformerSchema(BaseModelConfig):
    processor: GraphTransformerProcessorSchema = Field(default_factory=GraphTransformerProcessorSchema)
    "Graph transformer processor schema."
    encoder: GraphTransformerEncoderSchema = Field(default_factory=GraphTransformerEncoderSchema)
    "Graph transformer encoder schema."
    decoder: GraphTransformerDecoderSchema = Field(default_factory=GraphTransformerDecoderSchema)
    "Graph transformer decoder schema."


class TransformerSchema(BaseModelConfig):
    processor: TransformerProcessorSchema = Field(default_factory=TransformerProcessorSchema)
    "Transformer processor schema."
    encoder: GraphTransformerEncoderSchema = Field(default_factory=GraphTransformerEncoderSchema)
    "Graph transformer encoder schema."
    decoder: GraphTransformerDecoderSchema = Field(default_factory=GraphTransformerDecoderSchema)
    "Graph transformer decoder schema."
