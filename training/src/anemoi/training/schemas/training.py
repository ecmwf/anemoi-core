# (C) Copyright 2024-2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from enum import Enum
from functools import partial
from typing import Annotated
from typing import Any
from typing import Literal

from pydantic import AfterValidator
from pydantic import Discriminator
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import NonNegativeInt
from pydantic import PositiveInt
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import Self

from anemoi.utils.schemas import BaseModel
from anemoi.utils.schemas.errors import allowed_values


class GradientClip(BaseModel):
    """Gradient clipping configuration."""

    val: float = 32.0
    "Gradient clipping value."
    algorithm: Annotated[str, AfterValidator(partial(allowed_values, values=["value", "norm"]))] = Field(
        example="value",
    )
    "The gradient clipping algorithm to use"


class SWA(BaseModel):
    """Stochastic weight averaging configuration.

    See https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/
    """

    enabled: bool = Field(example=False)
    "Enable stochastic weight averaging."
    lr: NonNegativeFloat = Field(example=1.0e-4)
    "Learning rate for SWA."


class Rollout(BaseModel):
    """Rollout configuration."""

    start: PositiveInt = Field(example=1)
    "Number of rollouts to start with."
    epoch_increment: NonNegativeInt = Field(example=0)
    "Number of epochs to increment the rollout."
    max: PositiveInt = Field(example=1)
    "Maximum number of rollouts."


class LR(BaseModel):
    """Learning rate configuration.

    Changes in per-gpu batch_size should come with a rescaling of the local_lr,
    in order to keep a constant global_lr global_lr = local_lr * num_gpus_per_node * num_nodes / gpus_per_model.
    """

    rate: NonNegativeFloat = Field(example=0.625e-4)  # TODO(Helen): Could be computed by pydantic
    "Initial learning rate. Is adjusteed according to the hardware configuration"
    iterations: NonNegativeInt = Field(example=300000)
    "Number of iterations."
    min: NonNegativeFloat = Field(example=3e-7)
    "Minimum learning rate."
    warmup: NonNegativeInt = Field(example=1000)
    "Number of warm up iteration. Default to 1000."


class OptimizerSchema(BaseModel):
    """Optimizer configuration."""

    zero: bool = Field(example=False)
    "Use Zero optimiser."
    kwargs: dict[str, Any] = Field(default_factory=dict)
    "Additional arguments to pass to the optimizer."


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


class TendencyScalerTargets(str, Enum):
    stdev = "anemoi.training.losses.scalers.StdevTendencyScaler"
    var = "anemoi.training.losses.scalers.VarTendencyScaler"


class TendencyScalerSchema(BaseModel):
    target_: TendencyScalerTargets = Field(
        example="anemoi.training.losses.scalers.StdevTendencyScaler",
        alias="_target_",
    )


class VariableLevelScalerTargets(str, Enum):
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
    nodes_name: str = Field(example="data")
    "Name of the nodes to take the attribute from."
    nodes_attribute_name: str = Field(example="area_weight")
    "Name of the node attribute to return."
    norm: Literal["unit-max", "unit-sum"] | None = Field(example="unit-sum")
    "Normalisation method applied to the node attribute."


class ReweightedGraphNodeAttributeScalerSchema(BaseModel):
    target_: Literal["anemoi.training.losses.scalers.ReweightedGraphNodeAttributeScaler"] = Field(
        ...,
        alias="_target_",
    )
    nodes_name: str = Field(example="data")
    "Name of the nodes to take the attribute from."
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
    | ReweightedGraphNodeAttributeScalerSchema
)


class ImplementedLossesUsingBaseLossSchema(str, Enum):
    kcrps = "anemoi.training.losses.kcrps.KernelCRPS"
    afkcrps = "anemoi.training.losses.kcrps.AlmostFairKernelCRPS"
    rmse = "anemoi.training.losses.RMSELoss"
    mse = "anemoi.training.losses.MSELoss"
    weighted_mse = "anemoi.training.losses.WeightedMSELoss"
    mae = "anemoi.training.losses.MAELoss"
    logcosh = "anemoi.training.losses.LogCoshLoss"
    huber = "anemoi.training.losses.HuberLoss"


class BaseLossSchema(BaseModel):
    target_: ImplementedLossesUsingBaseLossSchema = Field(..., alias="_target_")
    "Loss function object from anemoi.training.losses."
    scalers: list[str] = Field(example=["variable"])  # TODO(Mario): Validate scalers are defined
    "Scalars to include in loss calculation"
    ignore_nans: bool = False
    "Allow nans in the loss and apply methods ignoring nans for measuring the loss."


class KernelCRPSSchema(BaseLossSchema):
    fair: bool = True
    "Calculate a 'fair' (unbiased) score - ensemble variance component weighted by (ens-size-1)^-1"


class AlmostFairKernelCRPSSchema(BaseLossSchema):
    alpha: float = 1.0
    """Factor for linear combination of fair (unbiased, ensemble variance component
    weighted by (ens-size-1)^-1) and standard CRPS (1.0 = fully fair, 0.0 = fully unfair)"""
    no_autocast: bool = True
    "Deactivate autocast for the kernel CRPS calculation"


class HuberLossSchema(BaseLossSchema):
    delta: float = 1.0
    "Threshold for Huber loss."


class CombinedLossSchema(BaseLossSchema):
    target_: Literal["anemoi.training.losses.combined.CombinedLoss"] = Field(..., alias="_target_")
    losses: list[BaseLossSchema] = Field(min_length=1)
    "Losses to combine, can be any of the normal losses."
    loss_weights: list[int | float] | None = None
    "Weightings of losses, if not set, all losses are weighted equally."

    @field_validator("losses", mode="before")
    @classmethod
    def add_empty_scalars(cls, losses: Any) -> Any:
        """Add empty scalars to loss functions, as scalars can be set at top level."""
        from omegaconf.omegaconf import open_dict

        for loss in losses:
            if "scalars" not in loss:
                with open_dict(loss):
                    loss["scalars"] = []
        return losses

    @model_validator(mode="after")
    def check_length_of_weights_and_losses(self) -> Self:
        """Check that the number of losses and weights match, or if not set, skip."""
        losses, loss_weights = self.losses, self.loss_weights
        if loss_weights is not None and len(losses) != len(loss_weights):
            error_msg = "Number of losses and weights must match"
            raise ValueError(error_msg)
        return self


LossSchemas = BaseLossSchema | HuberLossSchema | CombinedLossSchema | AlmostFairKernelCRPSSchema | KernelCRPSSchema


class ImplementedStrategiesUsingBaseDDPStrategySchema(str, Enum):
    ddp_ens = "anemoi.training.distributed.strategy.DDPEnsGroupStrategy"
    ddp = "anemoi.training.distributed.strategy.DDPGroupStrategy"


class BaseDDPStrategySchema(BaseModel):
    """Strategy configuration."""

    target_: ImplementedStrategiesUsingBaseDDPStrategySchema = Field(..., alias="_target_")
    num_gpus_per_model: PositiveInt = Field(example=2)
    "Number of GPUs per model."
    read_group_size: PositiveInt = Field(example=1)
    "Number of GPUs per reader group. Defaults to number of GPUs."


class DDPEnsGroupStrategyStrategySchema(BaseDDPStrategySchema):
    """Strategy object from anemoi.training.strategy."""

    num_gpus_per_ensemble: PositiveInt = Field(example=2)
    "Number of GPUs per ensemble."


StrategySchemas = BaseDDPStrategySchema | DDPEnsGroupStrategyStrategySchema


class BaseTrainingSchema(BaseModel):
    """Training configuration."""

    run_id: str | None = Field(example=None)
    "Run ID: used to resume a run from a checkpoint, either last.ckpt or specified in hardware.files.warm_start."
    fork_run_id: str | None = Field(example=None)
    "Run ID to fork from, either last.ckpt or specified in hardware.files.warm_start."
    load_weights_only: bool = Field(example=False)
    "Load only the weights from the checkpoint, not the optimiser state."
    transfer_learning: bool = Field(example=False)
    "Flag to activate transfer learning mode when loading a checkpoint."
    submodules_to_freeze: list[str] = Field(example=["processor"])
    "List of submodules to freeze during transfer learning."
    deterministic: bool = Field(default=False)
    "This flag sets the torch.backends.cudnn.deterministic flag. Might be slower, but ensures reproducibility."
    precision: str = Field(default="16-mixed")
    "Precision"
    multistep_input: PositiveInt = Field(example=2)
    """Number of input steps for the model. E.g. 1 = single step scheme, X(t-1) used to predict X(t),
    k > 1: multistep scheme, uses [X(t-k), X(t-k+1), ... X(t-1)] to predict X(t)."""
    accum_grad_batches: PositiveInt = Field(default=1)
    """Accumulates gradients over k batches before stepping the optimizer.
    K >= 1 (if K == 1 then no accumulation). The effective bacthsize becomes num-device * k."""
    num_sanity_val_steps: NonNegativeInt = Field(example=6)
    "Sanity check runs n batches of val before starting the training routine."
    gradient_clip: GradientClip
    "Config for gradient clipping."
    strategy: StrategySchemas
    "Strategy to use."
    swa: SWA = Field(default_factory=SWA)
    "Config for stochastic weight averaging."
    training_loss: LossSchemas
    "Training loss configuration."
    loss_gradient_scaling: bool = False
    "Dynamic rescaling of the loss gradient. Not yet tested."
    scalers: dict[str, ScalerSchema]
    "Scalers to use in the computation of the loss and validation scores."
    validation_metrics: dict[str, LossSchemas]
    "List of validation metrics configurations."
    variable_groups: dict[str, str | list[str] | dict[str, str | bool | list[str | int]]]
    "Groups for variable loss scaling"
    max_epochs: PositiveInt | None = None
    "Maximum number of epochs, stops earlier if max_steps is reached first."
    max_steps: PositiveInt = 150000
    "Maximum number of steps, stops earlier if max_epochs is reached first."
    lr: LR = Field(default_factory=LR)
    "Learning rate configuration."
    optimizer: OptimizerSchema = Field(default_factory=OptimizerSchema)
    "Optimizer configuration."
    metrics: list[str]
    "List of metrics"
    ensemble_size_per_device: PositiveInt = 1
    "Number of ensemble members per device. Default is 1 for non-ensemble forecasting."


class ForecasterSchema(BaseTrainingSchema):
    model_task: Literal["anemoi.training.train.tasks.GraphForecaster",] = Field(..., alias="model_task")
    "Training objective."
    rollout: Rollout = Field(default_factory=Rollout)
    "Rollout configuration."


class ForecasterEnsSchema(ForecasterSchema):
    model_task: Literal["anemoi.training.train.tasks.GraphEnsForecaster",] = Field(..., alias="model_task")
    "Training objective."


class DiffusionForecasterSchema(ForecasterSchema):
    model_task: Literal["anemoi.training.train.tasks.GraphDiffusionForecaster"] = Field(..., alias="model_task")
    "Training objective."


class DiffusionTendForecasterSchema(ForecasterSchema):
    model_task: Literal["anemoi.training.train.tasks.GraphDiffusionTendForecaster"] = Field(
        ...,
        alias="model_task",
    )
    "Training objective."


class InterpolationSchema(BaseTrainingSchema):
    model_task: Literal["anemoi.training.train.tasks.GraphInterpolator"] = Field(..., alias="model_task")
    "Training objective."
    explicit_times: ExplicitTimes
    "Time indices for input and output."
    target_forcing: TargetForcing
    "Forcing parameters for target output times."


TrainingSchema = Annotated[
    ForecasterSchema
    | ForecasterEnsSchema
    | InterpolationSchema
    | DiffusionForecasterSchema
    | DiffusionTendForecasterSchema,
    Discriminator("model_task"),
]
