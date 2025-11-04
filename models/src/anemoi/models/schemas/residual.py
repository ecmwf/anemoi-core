from typing import Literal

from pydantic import Field
from typing import Annotated

from anemoi.utils.schemas import BaseModel


class SkipConnectionSchema(BaseModel):
    """Schema for skip connection residuals."""
    target_: Literal[
        "anemoi.models.layers.residual.SkipConnection",
        "anemoi.models.layers.residual.NoConnection"
    ] = Field(
        default="anemoi.models.layers.residual.SkipConnection", alias="_target_"
    )


class TruncatedConnectionSchema(BaseModel):
    """Schema for truncated connection residuals."""
    target_: Literal["anemoi.models.layers.residual.TruncatedConnection"] = Field(..., alias="_target_")
    data_nodes: str = Field(
        ..., description="Name of the node set in the graph representing the original (full) resolution data."
    )
    truncation_nodes: str = Field(
        ..., description="Name of the node set in the graph representing the truncated (reduced) resolution data."
    )
    edge_weight_attribute: str | None = Field(
        None,
        description="Optional name of the edge attribute to use as weights when projecting between data and truncation nodes.",
    )
    src_node_weight_attribute: str | None = Field(
        None,
        description="Optional name of an attribute on source nodes to use as multiplicative weights during projection.",
    )
    autocast: bool = Field(
        False, description="Whether to enable mixed precision autocasting during projection operations."
    )


ResidualConnectionSchema = Annotated[
    SkipConnectionSchema | TruncatedConnectionSchema,
    Field(discriminator="target_"),
]