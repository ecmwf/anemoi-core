from typing import Annotated
from typing import Literal

from pydantic import Field

from anemoi.utils.schemas import BaseModel


class SkipConnectionSchema(BaseModel):
    """Schema for skip connection residuals."""

    target_: Literal["anemoi.models.layers.residual.SkipConnection"] = Field(..., alias="_target_")
    step: int = Field(
        -1,
        description="Timestep index to use for the skip connection. "
        "Defaults to -1, which selects the most recent timestep.",
    )


class TruncatedConnectionSchema(BaseModel):
    """Schema for truncated connection residuals."""

    target_: Literal["anemoi.models.layers.residual.TruncatedConnection"] = Field(..., alias="_target_")
    down_provider: str = Field(..., description="Name of projection provider for downscaling.")
    up_provider: str = Field(..., description="Name of projection provider for upscaling.")
    autocast: bool = Field(
        False, description="Whether to enable mixed precision autocasting during projection operations."
    )


ResidualConnectionSchema = Annotated[
    SkipConnectionSchema | TruncatedConnectionSchema,
    Field(discriminator="target_"),
]
