# (C) Copyright 2026 Anemoi contributors.
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
from pydantic import NonNegativeInt
from pydantic import PositiveInt
from pydantic import model_validator

from .common_components import TransformerModelComponent


class GraphTransformerRefinerSchema(TransformerModelComponent):
    target_: Literal["anemoi.models.layers.mapper.GraphTransformerRefinerMapper"] = Field(..., alias="_target_")
    "Graph Transformer refiner object from anemoi.models.layers.mapper."
    trainable_size: NonNegativeInt = Field(example=8)
    "Size of trainable parameters vector."
    sub_graph_edge_attributes: list[str] = Field(example=["edge_length", "edge_dirs"])
    "Edge attributes to consider in the refiner features."
    qk_norm: bool = Field(default=False)
    "Normalize the query and key vectors."
    attn_dim: PositiveInt | None = Field(default=None)
    "Optional attention dimension for q/k/v projections. Defaults to hidden_dim."
    refiner_channels: PositiveInt = Field(default=128)
    "Data-grid channel width used by the refinement stage."
    num_refiners: PositiveInt = Field(default=1)
    "Number of chained refiner mapper blocks."
    use_src_embedding: bool = Field(default=False)
    "Enable source embedding in refiner mappers."
    use_dst_embedding: bool = Field(default=False)
    "Enable destination embedding in refiner mappers."
    use_output_projection: bool | None = Field(default=None)
    "Optional override for output projection behavior inside each mapper block."

    @model_validator(mode="after")
    def check_valid_extras(self) -> Any:
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
