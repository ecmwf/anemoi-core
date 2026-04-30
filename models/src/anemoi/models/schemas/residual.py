from typing import Annotated
from typing import Literal
from typing import Self

from pydantic import Field
from pydantic import model_validator

from anemoi.utils.schemas import BaseModel


class SkipConnectionSchema(BaseModel):
    """Schema for skip connection residuals."""

    target_: Literal["anemoi.models.layers.residual.SkipConnection"] = Field(..., alias="_target_")
    step: int = Field(
        -1,
        description="Timestep index to use for the skip connection. "
        "Defaults to -1, which selects the most recent timestep.",
    )


class TruncationConfigDiskSchema(BaseModel):
    """File-based truncation config: projection matrices loaded from .npz files."""

    truncation_up_file_path: str
    truncation_down_file_path: str


class TruncationConfigOnTheFlySchema(BaseModel):
    """On-the-fly truncation config: truncation subgraph built from the main graph."""

    grid: str | None = None
    node_builder: dict | None = None
    num_nearest_neighbours: int = 3
    sigma: float = 1.0

    @model_validator(mode="after")
    def check_grid_or_node_builder(self) -> Self:
        if self.grid is None and self.node_builder is None:
            raise ValueError("TruncationConfigOnTheFlySchema requires either 'grid' or 'node_builder'.")
        return self


class TruncatedConnectionSchema(BaseModel):
    """Schema for truncated connection residuals."""

    target_: Literal["anemoi.models.layers.residual.TruncatedConnection"] = Field(..., alias="_target_")
    # Hydra merges `step` from the default SkipConnection config when _target_ is overridden; ignore it.
    step: int = Field(-1, exclude=True)
    truncation_config: TruncationConfigDiskSchema | TruncationConfigOnTheFlySchema | None = None
    edge_weight_attribute: str | None = None
    src_node_weight_attribute: str | None = None
    autocast: bool = False
    row_normalize: bool = False
    # Deprecated: pass inside truncation_config instead.
    truncation_up_file_path: str | None = None
    truncation_down_file_path: str | None = None


ResidualConnectionSchema = Annotated[
    SkipConnectionSchema | TruncatedConnectionSchema,
    Field(discriminator="target_"),
]
