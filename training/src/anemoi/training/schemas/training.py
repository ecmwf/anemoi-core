# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from enum import StrEnum
from functools import partial
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Self

from pydantic import AfterValidator
from pydantic import ConfigDict
from pydantic import Discriminator
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import NonNegativeInt
from pydantic import PositiveInt
from pydantic import Tag
from pydantic import field_validator
from pydantic import model_validator

from anemoi.training.schemas.schema_utils import DatasetDict
from anemoi.utils.schemas import BaseModel
from anemoi.utils.schemas.errors import allowed_values


class GenericSchema(BaseModel):
    """Generic Hydra instantiation schema with a required _target_ and arbitrary extra fields."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    target_: str = Field(alias="_target_")
    """Hydra target class or function to instantiate."""


class OptimizerSchema(GenericSchema):
    """Hydra instantiation config for a PyTorch optimizer."""


class LRSchedulerSchema(GenericSchema):
    """Hydra instantiation config for a learning rate scheduler."""


class PLSchedulerSchema(BaseModel):
    """PyTorch Lightning LRSchedulerConfig wrapper fields (interval, monitor, etc.)."""

    model_config = ConfigDict(extra="allow")

    interval: str = "step"
    """Interval at which Lightning calls lr_scheduler_step ('step' or 'epoch'). Defaults to 'step'."""


class GradientClip(BaseModel):
    """Gradient clipping configuration."""

    val: float = 32.0
    "Gradient clipping value."
    algorithm: Annotated[str, AfterValidator(partial(allowed_values, values=["value", "norm"]))] = Field(
        example="value",
    )
    "The gradient clipping algorithm to use"


class WeightAveragingSchema(GenericSchema):
    """Hydra instantiation config for a weight averaging callback (EMA or SWA).

    Example:
        weight_averaging:
          _target_: anemoi.training.diagnostics.callbacks.weight_averaging.EMAWeightAveraging
          decay: 0.999
          update_starting_at_step: 1000

    The stock ``pytorch_lightning.callbacks.*WeightAveraging`` classes also instantiate
    but pair parameters/buffers positionally; that is unsafe with anemoi imputers and
    updating loss scalers (a warning will be logged).
    """


class OptimizationSchema(BaseModel):
    """Optimizer and LR scheduler configuration."""

    lr: NonNegativeFloat = Field(example=0.625e-4)
    "Base learning rate per GPU. Scaled by hardware config at runtime."
    optimizer: OptimizerSchema
    """Hydra instantiation config for the optimizer."""
    lr_scheduler: LRSchedulerSchema | None = None
    """Hydra instantiation config for the LR scheduler. If None, no scheduler is used."""
    pl_lr_scheduler: PLSchedulerSchema = Field(default_factory=PLSchedulerSchema)
    """PyTorch Lightning LRSchedulerConfig wrapper fields (interval, monitor, etc.)."""


class ExplicitTimes(BaseModel):
    """Time indices for input and output.

    Starts at index 0. Input and output can overlap.
    """

    input: list[NonNegativeInt] = Field(examples=[0, 1])
    "Input time indices."
    target: list[NonNegativeInt] = Field(examples=[2])
    "Target time indices."


class TargetForcing(BaseModel):
    """Forcing parameters for target output times.

    Extra forcing parameters to use as input to distinguish between different target times.
    """

    data: list[str] = Field(examples=["insolation"])
    "List of forcing parameters to use as input to the model at the interpolated step."
    time_fraction: bool = Field(example=True)
    "Use target time as a fraction between input boundary times as input."


class LossScalingSchema(BaseModel):
    default: int = 1
    "Default scaling value applied to the variables loss. Default to 1."
    pl: dict[str, NonNegativeFloat]
    "Scaling value associated to each pressure level variable loss."
    sfc: dict[str, NonNegativeFloat]
    "Scaling value associated to each surface variable loss."


class GeneralVariableLossScalerSchema(BaseModel):
    target_: Literal["anemoi.training.losses.scalers.GeneralVariableLossScaler"] = Field(..., alias="_target_")
    weights: dict[str, float]
    "Weight of each variable."  # Check keys (variables) are read ???


class VariableMaskingScalerSchema(BaseModel):
    target_: Literal["anemoi.training.losses.scalers.VariableMaskingLossScaler"] = Field(..., alias="_target_")
    variables: list[str] = Field(defaultexample=["tp"])
    "Variables to compute the loss over."
    invert: bool = Field(examples=False)
    "Flag to invert the variable mask."


class NaNMaskScalerSchema(BaseModel):
    target_: Literal["anemoi.training.losses.scalers.NaNMaskScaler"] = Field(..., alias="_target_")
    use_processors_tendencies: bool = Field(default=False)
    "Flag to include processors for tendencies when building the loss mask."


class TendencyScalerTargets(StrEnum):
    stdev = "anemoi.training.losses.scalers.StdevTendencyScaler"
    var = "anemoi.training.losses.scalers.VarTendencyScaler"


class TendencyScalerSchema(BaseModel):
    target_: TendencyScalerTargets = Field(
        example="anemoi.training.losses.scalers.StdevTendencyScaler",
        alias="_target_",
    )
    timestep: str | None = Field(default=None, example="6h")
    "Timestep key used to select tendency statistics for scalers."


class VariableLevelScalerTargets(StrEnum):
    relu_scaler = "anemoi.training.losses.scalers.ReluVariableLevelScaler"
    linear_scaler = "anemoi.training.losses.scalers.LinearVariableLevelScaler"
    polynomial_sclaer = "anemoi.training.losses.scalers.PolynomialVariableLevelScaler"
    no_scaler = "anemoi.training.losses.scalers.NoVariableLevelScaler"


class VariableLevelScalerSchema(BaseModel):
    target_: VariableLevelScalerTargets = Field(
        example="anemoi.training.losses.scalers.ReluVariableLevelScaler",
        alias="_target_",
    )
    group: str = Field(example="pl")
    "Group of variables to scale."
    slope: float = Field(example=1.0)
    "Slope of scaling function."
    y_intercept: float = Field(example=0.001)
    "Y-axis shift of scaling function."


class GraphNodeAttributeScalerSchema(BaseModel):
    target_: Literal["anemoi.training.losses.scalers.GraphNodeAttributeScaler"] = Field(..., alias="_target_")
    nodes_attribute_name: str = Field(example="area_weight")
    "Name of the node attribute to return."
    norm: Literal["unit-max", "unit-sum"] | None = Field(example="unit-sum")
    "Normalisation method applied to the node attribute."


class TimeStepScalerSchema(BaseModel):
    target_: Literal["anemoi.training.losses.scalers.TimeStepScaler"] = Field(..., alias="_target_")
    norm: Literal["unit-max", "unit-sum"] | None = Field(default="unit-sum", example="unit-sum")
    "Normalisation method applied to the weights."
    weights: list[float] = Field(example=[1.0, 1.0])
    "Weights for each time step."


class UniformTimeStepScalerSchema(BaseModel):
    target_: Literal["anemoi.training.losses.scalers.UniformTimeStepScaler"] = Field(..., alias="_target_")


class LeadTimeDecayScalerSchema(BaseModel):
    target_: Literal["anemoi.training.losses.scalers.LeadTimeDecayScaler"] = Field(..., alias="_target_")
    output_lead_times: list[int] = Field(example=[0, 6, 12, 18, 24])
    "Lead times corresponding to each output step."
    decay_factor: float = Field(example=0.1)
    "Decay factor for the lead time weights."
    max_lead_time: int = Field(example=24)
    "Maximum lead time for decay calculation."
    decay_type: Literal["linear", "exponential"] | None = Field(default="linear", example="linear")
    "Type of decay to apply."
    inverse: bool | None = Field(default=False, example=False)
    "If true, weights increase with lead time."
    norm: Literal["unit-max", "unit-sum"] | None = Field(default="unit-sum", example="unit-sum")
    "Normalisation method applied to the weights."


class ReweightedGraphNodeAttributeScalerSchema(BaseModel):
    target_: Literal["anemoi.training.losses.scalers.ReweightedGraphNodeAttributeScaler"] = Field(
        ...,
        alias="_target_",
    )
    nodes_attribute_name: str = Field(example="area_weight")
    "Name of the node attribute to return."
    scaling_mask_attribute_name: str = Field(example="cutout_mask")
    "Name of the node attribute to use as a mask to reweight the reference values."
    weight_frac_of_total: float = Field(example=0.5)
    "Fraction of total weight to assign to nodes within the scaling mask. The remaining weight is distributed among "
    "nodes outside the mask."
    norm: Literal["unit-max", "unit-sum"] | None = Field(example="unit-sum")
    "Normalisation method applied to the node attribute."


ScalerSchema = (
    GeneralVariableLossScalerSchema
    | VariableLevelScalerSchema
    | VariableMaskingScalerSchema
    | TendencyScalerSchema
    | NaNMaskScalerSchema
    | GraphNodeAttributeScalerSchema
    | TimeStepScalerSchema
    | UniformTimeStepScalerSchema
    | LeadTimeDecayScalerSchema
    | ReweightedGraphNodeAttributeScalerSchema
)


class ImplementedLossesUsingBaseLossSchema(StrEnum):
    crps = "anemoi.training.losses.CRPS"
    rmse = "anemoi.training.losses.RMSELoss"
    mse = "anemoi.training.losses.MSELoss"
    discrete_h1 = "anemoi.training.losses.DiscreteSobolevH1Loss"
    gradient_meanl2 = "anemoi.training.losses.GradientMeanSquareLoss"
    gradient_meanl1 = "anemoi.training.losses.GradientMeanAbsLoss"
    gradient_maxl1 = "anemoi.training.losses.GradientMaxAbsLoss"
    weighted_mse = "anemoi.training.losses.WeightedMSELoss"
    mae = "anemoi.training.losses.MAELoss"
    logcosh = "anemoi.training.losses.LogCoshLoss"
    huber = "anemoi.training.losses.HuberLoss"
    fcl = "anemoi.training.losses.FourierCorrelationLoss"
    lsd = "anemoi.training.losses.LogSpectralDistance"
    logfft2d = "anemoi.training.losses.LogFFT2Distance"
    spectral_crps = "anemoi.training.losses.SpectralCRPSLoss"
    spectral_l2 = "anemoi.training.losses.SpectralL2Loss"
    spectral_amse = "anemoi.training.losses.SpectralAMSELoss"


class CheckVariablesCompatibilitySchema(BaseModel):
    """Options forwarded to ``Variable.check_compatibility`` / ``Variable.compatible``.

    Each field may be set to ``True`` to skip the check for all variables, or to a list
    of variable names to skip only those specific variables.
    """

    ignore_units: bool | list[str] = False
    """Ignore unit mismatches.  ``True`` skips all unit checks; a list of variable names
    skips only those variables."""
    ignore_processing_period: bool | list[str] = False
    """Ignore accumulation-period mismatches.  ``True`` skips all period checks; a list of
    variable names skips only those variables."""
    ignore_time_processing: bool | list[str] = False
    """Ignore time-processing type mismatches.  ``True`` skips all time-processing checks;
    a list of variable names skips only those variables."""
    ignore_type_of_level: bool | list[str] = False
    """Ignore mismatches in pressure level status.  ``True`` skips all type-of-level checks;
    a list of variable names skips only those variables."""


class BaseLossSchema(BaseModel):
    target_: ImplementedLossesUsingBaseLossSchema = Field(..., alias="_target_")
    "Loss function object from anemoi.training.losses."
    scalers: list[str] = Field(example=["variable"])  # TODO(Mario): Validate scalers are defined
    "Scalers to include in loss calculation"
    ignore_nans: bool = False
    "Allow nans in the loss and apply methods ignoring nans for measuring the loss."
    predicted_variables: list[str] | None = None
    target_variables: list[str] | None = None
    check_variables_compatibility: CheckVariablesCompatibilitySchema = Field(
        default_factory=CheckVariablesCompatibilitySchema,
    )
    "Options forwarded to ``Variable.check_compatibility`` when checking predicted vs. target variable units."


class CRPSSchema(BaseLossSchema):
    alpha: float = Field(default=0.95, ge=0.0, le=1.0)
    """Factor for linear combination of fair CRPS and standard CRPS.
    Values between 0 and 1 give the almost fair CRPS formulation."""
    backend: Literal["naive", "stable"] = "stable"
    "Backend used for the point-wise CRPS calculation."
    no_autocast: bool = True
    "Deactivate autocast for the kernel CRPS calculation"


class GradientLossSchema(BaseLossSchema):
    graph : str
    "Graph defining the knn from the data to the data"
    graph_name : str
    "Name for data grid"

class GraphLossMatrixSchema(BaseModel):
    """One graph-backed smoothing matrix definition for multiscale loss."""

    edges_name: tuple[str, str, str]
    edge_weight_attribute: str | None = None
    src_node_weight_attribute: str | None = None
    row_normalize: bool = False


class MultiscaleConfigDiskSchema(BaseModel):
    """File-based multiscale config: smoothing matrices loaded from .npz files."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    loss_matrices_path: str | None = None
    loss_matrices: list[str | None]
    scalers: list[str] | None = None
    "Scalers to apply to the wrapped loss (delegated to inner per_scale_loss)."


class MultiscaleConfigOnTheFlySchema(BaseModel):
    """On-the-fly multiscale config: smoothing subgraphs built from the main graph."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    num_scales: int | None = None
    base_num_nearest_neighbours: int | None = None
    base_sigma: float | None = None
    scale_factor: int | None = None
    smoothers: dict[str, dict] | None = None

    @model_validator(mode="after")
    def check_num_scales_or_smoothers(self) -> Self:
        if self.smoothers is not None:
            return self

        if self.num_scales is None:
            msg = "MultiscaleConfigOnTheFlySchema requires either 'num_scales' or 'smoothers'."
            raise ValueError(msg)

        missing = [name for name in ("base_num_nearest_neighbours", "base_sigma") if getattr(self, name) is None]
        if missing:
            msg = (
                "MultiscaleConfigOnTheFlySchema with 'num_scales' requires "
                f"{', '.join(repr(name) for name in missing)}."
            )
            raise ValueError(msg)
        return self


class TimeAggregateLossWrapperSchema(BaseModel):
    """Schema for TimeAggregateLossWrapper used inside CombinedLoss."""

    target_: Literal["anemoi.training.losses.aggregate.TimeAggregateLossWrapper"] = Field(..., alias="_target_")
    time_aggregation_types: list[Literal["diff", "mean", "min", "max"]] = Field(min_length=1)
    "Time aggregation operations to apply over the time dimension before computing the loss."
    loss_fn: BaseLossSchema | CRPSSchema
    "Inner loss function applied to each time-aggregated output."
    scalers: list[str] | None = None
    "Scalers to apply to the wrapped loss (delegated to inner loss_fn)."

    @field_validator("loss_fn", mode="before")
    @classmethod
    def add_empty_scalers_to_inner(cls, v: Any) -> Any:
        """Inject empty scalers for inner loss if missing; scalers flow through the wrapper.

        This is needed to avoid validation errors on the inner loss when scalers are only defined at the wrapper level.
        """
        if isinstance(v, dict) and "scalers" not in v:
            v["scalers"] = []
        else:
            from omegaconf import DictConfig
            from omegaconf.omegaconf import open_dict

            if isinstance(v, DictConfig) and "scalers" not in v:
                with open_dict(v):
                    v["scalers"] = []
        return v


class MultiScaleLossSchema(BaseModel):
    target_: Literal["anemoi.training.losses.MultiscaleLossWrapper"] = Field(..., alias="_target_")
    per_scale_loss: CRPSSchema | TimeAggregateLossWrapperSchema | BaseLossSchema
    weights: list[float]
    multiscale_config: MultiscaleConfigDiskSchema | MultiscaleConfigOnTheFlySchema | None = None
    sparse_projector_num_chunks: PositiveInt = 1
    # Deprecated: pass inside multiscale_config instead.
    loss_matrices_path: str | None = None
    loss_matrices: list[str | None] | None = None

    @field_validator("per_scale_loss", mode="before")
    @classmethod
    def add_empty_scalers_to_inner(cls, v: Any) -> Any:
        """Inject empty scalers for inner loss if missing; scalers flow through the wrapper.

        This is needed to avoid validation errors on the inner loss when scalers are only defined at the wrapper level.
        """
        if isinstance(v, dict) and "scalers" not in v:
            v["scalers"] = []
        else:
            from omegaconf import DictConfig
            from omegaconf.omegaconf import open_dict

            if isinstance(v, DictConfig) and "scalers" not in v:
                with open_dict(v):
                    v["scalers"] = []
        return v

    @model_validator(mode="after")
    def check_no_deprecated_mixed_with_on_the_fly(self) -> Self:
        if isinstance(self.multiscale_config, MultiscaleConfigOnTheFlySchema) and (
            self.loss_matrices is not None or self.loss_matrices_path is not None
        ):
            msg = (
                "Deprecated top-level 'loss_matrices'/'loss_matrices_path' must not be combined "
                "with an on-the-fly 'multiscale_config'. Move file-based keys inside a disk-mode "
                "multiscale_config, or remove the deprecated fields."
            )
            raise ValueError(msg)
        return self


class HuberLossSchema(BaseLossSchema):
    delta: float = 1.0
    "Threshold for Huber loss."


class SpectralProjectionConfigSchema(BaseModel):
    """Configuration for sparse projection applied before the spectral transform.

    Exactly one mode must be configured:

    - **File mode**: provide ``matrix_path``.
    - **Edge mode**: provide ``edges_name``.
    - **Target-grid mode**: provide ``num_nearest_neighbours`` and ``sigma``
      together with either ``grid`` or ``node_builder``.
    """

    model_config = ConfigDict(extra="forbid")

    # --- file mode ---
    matrix_path: str | None = None

    # --- edge mode ---
    edges_name: tuple[str, str, str] | None = None

    # --- target-grid mode ---
    num_nearest_neighbours: int | None = None
    sigma: float | None = None
    grid: str | None = None
    node_builder: dict | None = None
    target_node_name: str = "target_grid"

    # --- shared (edge and target-grid modes) ---
    edge_weight_attribute: str | None = None
    src_node_weight_attribute: str | None = None
    row_normalize: bool = False

    @model_validator(mode="after")
    def check_mode(self) -> Self:
        has_matrix = self.matrix_path is not None
        has_edges = self.edges_name is not None
        has_target_grid = self.num_nearest_neighbours is not None

        if has_matrix and has_edges:
            msg = "projection_config: 'matrix_path' and 'edges_name' are mutually exclusive"
            raise ValueError(msg)
        if has_matrix and has_target_grid:
            msg = "projection_config: 'matrix_path' and target-grid keys are mutually exclusive"
            raise ValueError(msg)
        if has_edges and has_target_grid:
            msg = "projection_config: 'edges_name' and target-grid keys are mutually exclusive"
            raise ValueError(msg)
        if not (has_matrix or has_edges or has_target_grid):
            msg = "projection_config must specify 'matrix_path', 'edges_name', or target-grid keys"
            raise ValueError(msg)
        if has_target_grid and self.sigma is None:
            msg = "projection_config: target-grid mode requires 'sigma'"
            raise ValueError(msg)
        if has_target_grid and self.grid is None and self.node_builder is None:
            msg = "projection_config: target-grid mode requires 'grid' or 'node_builder'"
            raise ValueError(msg)
        return self


class SpectralLossSchema(BaseLossSchema):
    """Spectral loss class."""

    transform: Literal["fft2d", "dct2d", "reduced_sht", "octahedral_sht"] = Field(..., example="fft2d")
    """Type of spectral transform to use."""
    subgrid: tuple[int, int | None] | str | None = None
    """Optional slice or string to select a subgrid before the transform."""
    projection_config: SpectralProjectionConfigSchema | None = None
    """Optional sparse projection applied to the data before the spectral transform."""

    @model_validator(mode="after")
    def check_subgrid_transform(self) -> Self:
        if self.subgrid is not None and self.transform in ("reduced_sht", "octahedral_sht"):
            msg = (
                f"subgrid is not supported for the '{self.transform}' transform: "
                "spherical harmonic transforms require the full grid"
            )
            raise ValueError(msg)
        return self

    class Config(BaseModel.Config):
        """Override to allow extra parameters for spectral transforms."""

        extra = "allow"


def _loss_discriminator(v: Any) -> str:
    target = v.get("_target_", "") if hasattr(v, "get") else getattr(v, "target_", "")
    if target == "anemoi.training.losses.combined.CombinedLoss":
        return "combined"
    if target == "anemoi.training.losses.MultiscaleLossWrapper":
        return "multiscale"
    if target == "anemoi.training.losses.CRPS":
        return "crps"
    if target in {
        "anemoi.training.losses.FourierCorrelationLoss",
        "anemoi.training.losses.LogSpectralDistance",
        "anemoi.training.losses.LogFFT2Distance",
        "anemoi.training.losses.SpectralCRPSLoss",
        "anemoi.training.losses.SpectralL2Loss",
        "anemoi.training.losses.SpectralAMSELoss",
    }:
        return "spectral"
    if target in {
        "anemoi.training.losses.DiscreteSobolevH1Loss",
        "anemoi.training.losses.GradientMeanSquareLoss",
        "anemoi.training.losses.GradientMeanAbsLoss",
        "anemoi.training.losses.GradientMaxAbsLoss",
    }:
        return "gradient"
    if target == "anemoi.training.losses.HuberLoss":
        return "huber"
    if target == "anemoi.training.losses.aggregate.TimeAggregateLossWrapper":
        return "time_aggregate"
    return "base"


class CombinedLossSchema(BaseLossSchema):
    """Schema for CombinedLoss.

    Top-level ``scalers`` act as defaults for sub-losses that don't specify their own.
    Sub-losses that explicitly set ``scalers`` override the parent value.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    target_: Literal["anemoi.training.losses.combined.CombinedLoss"] = Field(..., alias="_target_")
    "CombinedLoss target."
    scalers: list[str] = Field(default_factory=list, example=["variable"])
    "Optional top-level scalers propagated to sub-losses that don't define their own."
    losses: list[
        Annotated[
            Annotated[BaseLossSchema, Tag("base")]
            | Annotated[GradientLossSchema, Tag("gradient")]
            | Annotated[HuberLossSchema, Tag("huber")]
            | Annotated[CRPSSchema, Tag("crps")]
            | Annotated[SpectralLossSchema, Tag("spectral")]
            | Annotated[MultiScaleLossSchema, Tag("multiscale")]
            | Annotated[TimeAggregateLossWrapperSchema, Tag("time_aggregate")],
            Discriminator(_loss_discriminator),
        ]
    ] = Field(min_length=1)
    "Losses to combine, can be any of the normal losses."
    loss_weights: list[int | float] | None = None
    "Weightings of losses, if not set, all losses are weighted equally."

    @model_validator(mode="before")
    @classmethod
    def propagate_scalers_to_children(cls, data: Any) -> Any:
        """Propagate parent scalers to sub-losses that don't specify their own.

        MultiscaleLossWrapper is skipped because it manages scalers via per_scale_loss.
        """
        from omegaconf import DictConfig
        from omegaconf.omegaconf import open_dict

        parent_scalers = data.get("scalers", []) if hasattr(data, "get") else []
        if not parent_scalers:
            return data

        losses = data.get("losses", []) if hasattr(data, "get") else []
        for loss in losses:
            if not hasattr(loss, "get"):
                continue
            target = loss.get("_target_", "")
            # MultiscaleLossWrapper manages scalers on per_scale_loss, not at top level
            if "MultiscaleLossWrapper" in str(target):
                continue
            if "scalers" not in loss:
                if isinstance(loss, DictConfig):
                    with open_dict(loss):
                        loss["scalers"] = list(parent_scalers)
                elif isinstance(loss, dict):
                    loss["scalers"] = list(parent_scalers)
        return data

    @model_validator(mode="after")
    def check_length_of_weights_and_losses(self) -> Self:
        """Check that the number of losses and weights match, or if not set, skip."""
        losses, loss_weights = self.losses, self.loss_weights
        if loss_weights is not None and len(losses) != len(loss_weights):
            error_msg = "Number of losses and weights must match"
            raise ValueError(error_msg)
        return self


LossSchemas = Annotated[
    Annotated[BaseLossSchema, Tag("base")]
    | Annotated[GradientLossSchema, Tag("gradient")]
    | Annotated[HuberLossSchema, Tag("huber")]
    | Annotated[CombinedLossSchema, Tag("combined")]
    | Annotated[CRPSSchema, Tag("crps")]
    | Annotated[SpectralLossSchema, Tag("spectral")]
    | Annotated[TimeAggregateLossWrapperSchema, Tag("time_aggregate")]
    | Annotated[MultiScaleLossSchema, Tag("multiscale")],
    Discriminator(_loss_discriminator),
]


class ImplementedStrategiesUsingBaseDDPStrategySchema(StrEnum):
    ddp_ens = "anemoi.training.distributed.strategy.DDPEnsGroupStrategy"
    ddp = "anemoi.training.distributed.strategy.DDPGroupStrategy"


class BaseDDPStrategySchema(BaseModel):
    """Strategy configuration."""

    target_: ImplementedStrategiesUsingBaseDDPStrategySchema = Field(..., alias="_target_")
    num_gpus_per_model: PositiveInt = Field(example=2)
    "Number of GPUs per model."
    read_group_size: PositiveInt = Field(example=1)
    "Number of GPUs per reader group. Defaults to number of GPUs."
    use_local_synchronization: bool = Field(default=True, example=True)
    "Use synchronization local to the group when creating process groups."


class DDPEnsGroupStrategyStrategySchema(BaseDDPStrategySchema):
    """Strategy object from anemoi.training.strategy."""

    num_gpus_per_ensemble: PositiveInt = Field(example=2)
    "Number of GPUs per ensemble."


StrategySchemas = BaseDDPStrategySchema | DDPEnsGroupStrategyStrategySchema

VariableGroupType = dict[str, str | list[str] | dict[str, str | bool | list[str | int]]] | None


class UpdateDsStatsOnCkptLoadSchema(BaseModel):
    """Configuration for updating processor statistics on checkpoint load."""

    states: bool = Field(default=False, example=False)
    "Rebuild state pre/post-processing statistics from the current dataset."
    tendencies: bool = Field(default=True, example=True)
    "Rebuild tendency pre/post-processing statistics from the current dataset."


class BaseTrainingSchema(BaseModel):
    """Training configuration."""

    run_id: str | None = Field(example=None)
    "Run ID: used to resume a run from a checkpoint, either last.ckpt or specified in system.input.warm_start."
    fork_run_id: str | None = Field(example=None)
    "Run ID to fork from, either last.ckpt or specified in system.input.warm_start."
    load_weights_only: bool = Field(example=False)
    "Load only the weights from the checkpoint, not the optimiser state."
    transfer_learning: bool = Field(example=False)
    "Flag to activate transfer learning mode when loading a checkpoint."
    check_variables_compatibility: CheckVariablesCompatibilitySchema = Field(
        default_factory=CheckVariablesCompatibilitySchema,
    )
    "Options forwarded to ``Variable.check_compatibility`` when checking checkpoint vs. current dataset (fine-tuning)."
    update_ds_stats_on_ckpt_load: UpdateDsStatsOnCkptLoadSchema = Field(default_factory=UpdateDsStatsOnCkptLoadSchema)
    "Rebuild pre/post-processing statistics from the current dataset when loading a checkpoint."
    submodules_to_freeze: list[str] = Field(example=["processor"])
    "List of submodules to freeze during transfer learning."
    deterministic: bool = Field(default=False)
    "This flag sets torch.backends.cudnn.deterministic. It may reduce nondeterminism, but does not guarantee exact"
    " reproducibility."
    precision: str = Field(default="16-mixed")
    "Precision"
    preferred_blas_backend: str | None = Field(default=None)
    "Optionally override PyTorch's default BLAS backend."
    accum_grad_batches: PositiveInt = Field(default=1)
    """Accumulates gradients over k batches before stepping the optimizer.
    K >= 1 (if K == 1 then no accumulation). The effective bacthsize becomes num-device * k."""
    num_sanity_val_steps: NonNegativeInt = Field(example=6)
    "Sanity check runs n batches of val before starting the training routine."
    gradient_clip: GradientClip
    "Config for gradient clipping."
    strategy: StrategySchemas
    "Strategy to use."
    training_loss: DatasetDict[LossSchemas]
    "Training loss configuration."
    weight_averaging: WeightAveragingSchema | None = Field(default=None)
    "Config for weight averaging (SWA or EMA). Set to null to disable."
    loss_gradient_scaling: bool = False
    "Dynamic rescaling of the loss gradient. Not yet tested."
    scalers: DatasetDict[dict[str, ScalerSchema]]
    "Scalers to use in the computation of the loss and validation scores."
    validation_metrics: DatasetDict[dict[str, LossSchemas] | None]
    "List of validation metrics configurations."
    variable_groups: DatasetDict[VariableGroupType]
    "Groups for variable loss scaling"
    max_epochs: PositiveInt | None = None
    "Maximum number of epochs, stops earlier if max_steps is reached first."
    max_steps: PositiveInt = 150000
    "Maximum number of steps, stops earlier if max_epochs is reached first."
    optimization: OptimizationSchema
    "Optimizer and LR scheduler configuration."
    metrics: DatasetDict[list[str]]
    "List of metrics"
    ensemble_size_per_device: PositiveInt = 1
    "Number of ensemble members per device. Default is 1 for non-ensemble forecasting."


class SingleTrainingMethodSchema(BaseModel):
    target_: Literal["anemoi.training.train.methods.SingleTraining"] = Field(..., alias="_target_")
    "Hydra target for the single training method."


class EnsembleTrainingMethodSchema(BaseModel):
    target_: Literal["anemoi.training.train.methods.EnsembleTraining"] = Field(..., alias="_target_")
    "Hydra target for the ensemble training method."


class TransportTrainingMethodSchema(BaseModel):
    target_: Literal["anemoi.training.train.methods.TransportTraining"] = Field(..., alias="_target_")
    "Hydra target for the transport training method."


def _training_method_discriminator(v: Any) -> str:
    method = v.get("method", {}) if hasattr(v, "get") else getattr(v, "method", None)
    return method.get("_target_", "") if hasattr(method, "get") else getattr(method, "target_", "")


class SingleTrainingSchema(BaseTrainingSchema):
    method: SingleTrainingMethodSchema
    "Training method."


class EnsembleTrainingSchema(BaseTrainingSchema):
    method: EnsembleTrainingMethodSchema
    "Training method."


class TransportTrainingConfigSchema(BaseModel):
    prediction_mode: Literal["state", "tendency"] = "state"
    "Endpoint semantics for the transport objective."
    objective: Literal["edm_diffusion", "stochastic_interpolant"] = "edm_diffusion"
    "Transport objective used to perturb targets and train the model."


class TransportTrainingSchema(BaseTrainingSchema):
    method: TransportTrainingMethodSchema
    "Training method."
    transport: TransportTrainingConfigSchema = Field(default_factory=TransportTrainingConfigSchema)
    "Transport training configuration."


TrainingSchema = Annotated[
    Annotated[SingleTrainingSchema, Tag("anemoi.training.train.methods.SingleTraining")]
    | Annotated[EnsembleTrainingSchema, Tag("anemoi.training.train.methods.EnsembleTraining")]
    | Annotated[TransportTrainingSchema, Tag("anemoi.training.train.methods.TransportTraining")],
    Discriminator(_training_method_discriminator),
]
