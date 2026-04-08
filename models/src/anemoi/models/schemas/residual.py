from typing import Annotated
from typing import Any
from typing import Literal

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


class TruncatedConnectionSchema(BaseModel):
    """Schema for truncated connection residuals.

    Supports two modes:
    - **Config-based** (preferred): provide ``truncation_config`` with a ``grid``
      (or ``node_builder``) spec.  The truncation subgraph is built internally.
    - **File-based**: provide ``truncation_up_file_path`` and
      ``truncation_down_file_path`` to load pre-computed projection matrices.
    """

    target_: Literal["anemoi.models.layers.residual.TruncatedConnection"] = Field(..., alias="_target_")
    truncation_config: dict | None = Field(
        None,
        description="Truncation projection config (grid, num_nearest_neighbours, "
        "edge_weight_attribute, sigma, gaussian_norm). The truncation subgraph is "
        "built internally from this spec and the data-node positions in graph_data.",
    )
    edge_weight_attribute: str | None = Field(
        None,
        description="Override edge weight attribute name (defaults to value in truncation_config or 'gauss_weight').",
    )
    src_node_weight_attribute: str | None = Field(
        None,
        description="Optional source-node attribute to multiply into edge weights.",
    )
    truncation_up_file_path: str | None = Field(
        None,
        description="File path (.npz) for the up-projection matrix (file-based mode).",
    )
    truncation_down_file_path: str | None = Field(
        None,
        description="File path (.npz) for the down-projection matrix (file-based mode).",
    )
    autocast: bool = Field(False, description="Enable mixed-precision autocasting during projection.")
    row_normalize: bool = Field(False, description="Normalize projection matrix rows so each sums to 1.")

    @model_validator(mode="after")
    def check_instantiation_method(self) -> Any:
        file_based = self.truncation_up_file_path is not None and self.truncation_down_file_path is not None
        config_based = self.truncation_config is not None
        if file_based and config_based:
            raise ValueError("Specify either truncation_config or file paths, not both.")
        return self


ResidualConnectionSchema = Annotated[
    SkipConnectionSchema | TruncatedConnectionSchema,
    Field(discriminator="target_"),
]
