from typing import Literal

from pydantic import Field

from anemoi.utils.schemas import BaseModel


class SkipConnectionSchema(BaseModel):
    target_: Literal["anemoi.models.layers.residual.SkipConnection"] = Field(
        default="anemoi.models.layers.residual.SkipConnection", alias="_target_"
    )


class TruncationMapperSchema(BaseModel):
    target_: Literal["anemoi.models.layers.residual.TruncationMapper"] = Field(..., alias="_target_")


class NoConnectionSchema(BaseModel):
    target_: Literal["anemoi.models.layers.residual.NoConnection"] = Field(..., alias="_target_")
