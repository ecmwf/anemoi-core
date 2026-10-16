# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from typing import Annotated
from typing import Literal

from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import PositiveInt

from anemoi.utils.schemas import BaseModel


class SumAggregatorSchema(BaseModel):
    target_: Literal["anemoi.models.layers.aggregator.SumAggregator"] = Field(..., alias="_target_")


class MeanAggregatorSchema(BaseModel):
    target_: Literal["anemoi.models.layers.aggregator.MeanAggregator"] = Field(..., alias="_target_")


class ConcatAggregatorSchema(BaseModel):
    target_: Literal["anemoi.models.layers.aggregator.ConcatAggregator"] = Field(..., alias="_target_")


class PointwiseCrossAttentionAggregatorSchema(BaseModel):
    target_: Literal["anemoi.models.layers.aggregator.PointwiseCrossAttentionAggregator"] = Field(..., alias="_target_")
    num_channels: PositiveInt
    num_heads: PositiveInt
    layer_kernels: dict[str, dict] | None = Field(default_factory=dict)
    attn_channels: PositiveInt | None = None
    dropout_p: NonNegativeFloat = Field(default=0.0, le=1.0)
    qkv_bias: bool = False
    qk_norm: bool = False
    gradient_checkpointing: bool = True


AggregatorSchema = Annotated[
    SumAggregatorSchema | MeanAggregatorSchema | ConcatAggregatorSchema | PointwiseCrossAttentionAggregatorSchema,
    Field(discriminator="target_"),
]
