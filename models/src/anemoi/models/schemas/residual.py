# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from enum import Enum
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
    truncation_down_edges_name: tuple[str, str, str] | None = None
    truncation_up_edges_name: tuple[str, str, str] | None = None
    data_node_name: str | None = None
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
    regressors: list[str] | None = Field(
        None,
        description="Variable names to use as regressors.",
    )


class SpectralOrnsteinSupportedGrids(str, Enum):
    """Supported grid types for SpectralOrnsteinConnection."""

    REGULAR = "regular"
    OCTAHEDRAL = "octahedral"


class SpectralOrnsteinConnectionSchema(BaseModel):
    """Schema for spectral Ornstein residual connections."""

    target_: Literal["anemoi.models.layers.residual.SpectralOrnsteinConnection"] = Field(..., alias="_target_")
    lmax: int = Field(
        2,
        description="Maximum spherical harmonic degree for the theta/mu coefficients.",
    )
    grid: SpectralOrnsteinSupportedGrids = Field(
        SpectralOrnsteinSupportedGrids.REGULAR,
        description='Grid type: "regular" for regular lat-lon, "octahedral" for octahedral reduced grids.',
    )
    theta_init: float = Field(
        0.0,
        description="Initial value for theta.",
    )
    theta_buff: float = Field(
        0.0,
        description="Lower bound buffer for theta.",
    )
    use_mean: bool = Field(
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


class NoSkipConnectionSchema(BaseModel):
    """Schema for disabled residual connections (zeros)."""

    target_: Literal["anemoi.models.layers.residual.NoSkipConnection"] = Field(..., alias="_target_")
    # Hydra merges `step` from the default SkipConnection config when _target_ is overridden; ignore it.
    step: int = Field(-1, exclude=True)


class ClimatologySkipConnectionSchema(BaseModel):
    """Schema for climatology-based skip connections."""

    target_: Literal["anemoi.models.layers.residual.ClimatologySkipConnection"] = Field(..., alias="_target_")
    # Hydra merges `step` from the default SkipConnection config when _target_ is overridden; ignore it.
    step: int = Field(-1, exclude=True)
    climatology_path: str = Field(
        ...,
        description="Path to a .npz file mapping variable names to 1-D per-grid-point climatology arrays.",
    )
    missing_value: float = Field(
        0.0,
        description="Sentinel value marking missing points when fill_missing_only is enabled.",
    )
    fill_missing_only: bool = Field(
        False,
        description="If true, return the latest input with missing points replaced by climatology "
        "instead of returning the climatology everywhere.",
    )
    normalize_climatology: bool = Field(
        False,
        description="If true, normalize the loaded climatology with dataset statistics before use.",
    )
    normalize_method: dict | None = Field(
        None,
        description="Normalization method config, same structure as the preprocessor normalizer "
        '(e.g. {"default": "mean-std", "min-max": ["tp"]}). Only used when normalize_climatology is true.',
    )


# Leaf residual schemas usable as sub-residuals inside PerVariableGroupResidual.
SubResidualConnectionSchema = Annotated[
    SkipConnectionSchema
    | NoSkipConnectionSchema
    | ClimatologySkipConnectionSchema
    | TruncatedConnectionSchema
    | ScalarOrnsteinConnectionSchema
    | SpectralOrnsteinConnectionSchema,
    Field(discriminator="target_"),
]


class ResidualGroupSchema(BaseModel):
    """Schema for one variable group of a PerVariableGroupResidual."""

    name: str | None = Field(None, description="Group name, used in error messages.")
    variables: list[str] = Field(..., min_length=1, description="Model-input variable names in this group.")
    residual: SubResidualConnectionSchema = Field(..., description="Sub-residual applied to this group.")


class PerVariableGroupResidualSchema(BaseModel):
    """Schema for per-variable-group residual connections."""

    target_: Literal["anemoi.models.layers.residual.PerVariableGroupResidual"] = Field(..., alias="_target_")
    # Hydra merges `step` from the default SkipConnection config when _target_ is overridden; ignore it.
    step: int = Field(-1, exclude=True)
    groups: list[ResidualGroupSchema] = Field(..., min_length=1)


ResidualConnectionSchema = Annotated[
    SkipConnectionSchema
    | NoSkipConnectionSchema
    | ClimatologySkipConnectionSchema
    | PerVariableGroupResidualSchema
    | TruncatedConnectionSchema
    | ScalarOrnsteinConnectionSchema
    | SpectralOrnsteinConnectionSchema,
    Field(discriminator="target_"),
]
