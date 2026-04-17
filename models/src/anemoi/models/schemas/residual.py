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

    Supports two modes, both specified via ``truncation_config``:

    - **On-the-fly**: provide ``grid`` (or ``node_builder``) — truncation subgraph
      is built internally from the main graph::

        truncation_config:
          grid: o32
          num_nearest_neighbours: 3
          sigma: 1.0

    - **File-based**: provide ``truncation_up_file_path`` and
      ``truncation_down_file_path`` inside ``truncation_config``::

        truncation_config:
          truncation_down_file_path: /path/to/down.npz
          truncation_up_file_path: /path/to/up.npz
    """

    target_: Literal["anemoi.models.layers.residual.TruncatedConnection"] = Field(..., alias="_target_")
    truncation_config: dict | None = Field(
        None,
        description="Truncation config. For on-the-fly mode provide 'grid' (or 'node_builder'). "
        "For file mode provide 'truncation_up_file_path' and 'truncation_down_file_path'.",
    )
    src_node_weight_attribute: str | None = Field(None)
    autocast: bool = Field(False)
    row_normalize: bool = Field(False)
    # Deprecated: pass inside truncation_config instead.
    truncation_up_file_path: str | None = Field(None)
    truncation_down_file_path: str | None = Field(None)


ResidualConnectionSchema = Annotated[
    SkipConnectionSchema | TruncatedConnectionSchema,
    Field(discriminator="target_"),
]
