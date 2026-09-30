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
from typing import Optional
from typing import Union

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveFloat
from pydantic import PositiveInt
from pydantic import field_validator

from anemoi.utils.schemas import BaseModel


class NeighbourhoodSchema(BaseModel):
    grid: str = Field(example="octahedral")
    "Grid family of the nodes, a key of anemoi.models.layers.neighbourhood_attention.GRID_KERNELS, e.g. 'octahedral' or 'healpix'."
    kernel_size: tuple[PositiveInt, PositiveInt] = Field(example=(7, 13))
    "Latitude rows and points per row each query attends to; both odd."
    backend: Literal["triton", "flex", "sdpa"] = Field(default="triton")
    "Kernels to use: 'triton' (GPU), 'flex' (flex attention) or 'sdpa' (dense mask, small grids). Default to 'triton'."
    rotary_max_frequency: Optional[float] = Field(default=None, ge=1.0, example=100.0)
    "Rotary position embeddings from the 3D positions of the points, with frequencies from 1 to this value in radians per Earth radius (about pi over the grid spacing in radians, e.g. 100 for O48). Off when null."

    @field_validator("grid")
    @classmethod
    def check_grid(cls, grid: str) -> str:
        from anemoi.models.layers.neighbourhood_attention import GRID_KERNELS

        if grid not in GRID_KERNELS:
            raise ValueError(
                f"No neighbourhood attention kernels for grid '{grid}'. Known grids: {list(GRID_KERNELS)}."
            )
        return grid

    @field_validator("kernel_size")
    @classmethod
    def check_kernel_size(cls, kernel_size: tuple[int, int]) -> tuple[int, int]:
        if any(k % 2 == 0 for k in kernel_size):
            raise ValueError(f"kernel_size entries must be odd, got {kernel_size}.")
        return kernel_size


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
    if component.softcap or component.use_alibi_slopes:
        raise ValueError("Neighbourhood attention supports neither softcap nor alibi slopes.")
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
