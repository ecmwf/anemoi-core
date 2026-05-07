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
            msg = "TruncationConfigOnTheFlySchema requires either 'grid' or 'node_builder'."
            raise ValueError(msg)
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


class ScalarOrnsteinConnectionSchema(BaseModel):
    """Schema for scalar Ornstein residual connections."""

    target_: Literal["anemoi.models.layers.residual.ScalarOrnsteinConnection"] = Field(..., alias="_target_")
    theta_init: float = Field(
        0.0,
        description="Initial value for theta. If 0 and statistics are available, auto-initialized from tendency statistics.",
    )
    theta_buff: float = Field(
        0.0,
        description="Lower bound buffer for theta. Theta is constrained to (theta_buff, 1).",
    )
    theta_train: bool = Field(
        True,
        description="Whether theta is a trainable parameter.",
    )


class SpectralOrnsteinConnectionSchema(BaseModel):
    """Schema for spectral Ornstein residual connections."""

    target_: Literal["anemoi.models.layers.residual.SpectralOrnsteinConnection"] = Field(..., alias="_target_")
    lmax: int = Field(
        2,
        description="Maximum spherical harmonic degree for the theta/mu coefficients.",
    )
    grid: str = Field(
        "legendre-gauss",
        description='Grid type: "legendre-gauss" for regular lat-lon, "octahedral" for octahedral reduced grids.',
    )
    theta_init: float = Field(
        0.0,
        description="Initial value for theta.",
    )
    theta_buff: float = Field(
        0.0,
        description="Lower bound buffer for theta.",
    )
    zmean_term: bool = Field(
        True,
        description="Whether to include a zonal mean (mu) term.",
    )
    regressors: list[str] | None = Field(
        None,
        description="Variable names to use as spatially-varying regressors.",
    )
    truncate: bool = Field(
        False,
        description="If True, apply a learnable spectral low-pass filter to the input fields.",
    )
    anti_aliasing: bool = Field(
        True,
        description="If True (and truncate=True), use anti-aliasing blending in the filter.",
    )
    skip_truncate_variables: list[str] | None = Field(
        None,
        description="Variable names to exclude from spectral truncation (only used when truncate=True).",
    )


ResidualConnectionSchema = Annotated[
    SkipConnectionSchema
    | TruncatedConnectionSchema
    | ScalarOrnsteinConnectionSchema
    | SpectralOrnsteinConnectionSchema,
    Field(discriminator="target_"),
]
