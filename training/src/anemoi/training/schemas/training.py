# (C) Copyright 2024-2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

from enum import Enum
from functools import partial
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Union

from pydantic import AfterValidator
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import NonNegativeInt
from pydantic import PositiveInt
from pydantic import field_validator
from pydantic import model_validator

from anemoi.training.schemas.utils import allowed_values

from .utils import BaseModel


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
    warmup_t: NonNegativeInt = Field(example=1000)
    "Number of warm up iteration. Default to 1000."


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


class PressureLevelScalerTargets(str, Enum):

    relu_scaler = "anemoi.training.data.scaling.ReluPressureLevelScaler"
    linear_scaler = "anemoi.training.data.scaling.LinearPressureLevelScaler"
    polynomial_sclaer = "anemoi.training.data.scaling.PolynomialPressureLevelScaler"
    no_scaler = "anemoi.training.data.scaling.NoPressureLevelScaler"


class PressureLevelScalerSchema(BaseModel):
    target_: PressureLevelScalerTargets = Field(
        example="anemoi.training.data.scaling.ReluPressureLevelScaler",
        alias="_target_",
    )
    minimum: float = Field(example=0.2)
    "Minimum value of the scaling function."
    slope: float = 0.001
    "Slope of the scaling function."


PossibleScalars = Annotated[
    str,
    AfterValidator(partial(allowed_values, values=["limited_area_mask", "variable", "loss_weights_mask", "*"])),
]


class ImplementedLossesUsingBaseLossSchema(str, Enum):
    rmse = "anemoi.training.losses.rmse.WeightedRMSELoss"
    mse = "anemoi.training.losses.mse.WeightedMSELoss"
    mae = "anemoi.training.losses.mae.WeightedMAELoss"
    logcosh = "anemoi.training.losses.logcosh.WeightedLogCoshLoss"
    huber = "anemoi.training.losses.huber.WeightedHuberLoss"
    limited_mse = "anemoi.training.losses.limitedarea.WeightedMSELossLimitedArea"


class BaseLossSchema(BaseModel):
    target_: ImplementedLossesUsingBaseLossSchema = Field(..., alias="_target_")
    "Loss function object from anemoi.training.losses."
    scalars: list[PossibleScalars] = Field(example=["variable"])
    "Scalars to include in loss calculation"
    ignore_nans: bool = False
    "Allow nans in the loss and apply methods ignoring nans for measuring the loss."


class HuberLossSchema(BaseLossSchema):
    delta: float = 1.0
    "Threshold for Huber loss."


class WeightedMSELossLimitedAreaSchema(BaseLossSchema):
    inside_lam: bool = True
    "Whether to compute the MSE inside or outside the limited area."
    wmse_contribution: bool = False
    "Whether to compute the contribution to the MSE or not."


class CombinedLossSchema(BaseLossSchema):
    target_: Literal["anemoi.training.losses.combined.CombinedLoss"] = Field(..., alias="_target_")
    losses: list[BaseLossSchema] = Field(min_length=1)
    "Losses to combine, can be any of the normal losses."
    loss_weights: Union[list[Union[int, float]], None] = None
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
    def check_length_of_weights_and_losses(self) -> CombinedLossSchema:
        """Check that the number of losses and weights match, or if not set, skip."""
        losses, loss_weights = self.losses, self.loss_weights
        if loss_weights is not None and len(losses) != len(loss_weights):
            error_msg = "Number of losses and weights must match"
            raise ValueError(error_msg)
        return self


LossSchemas = Union[BaseLossSchema, HuberLossSchema, WeightedMSELossLimitedAreaSchema, CombinedLossSchema]


class GraphNodeAttributeSchema(BaseModel):
    target_: Literal["anemoi.training.losses.nodeweights.GraphNodeAttribute"] = Field(..., alias="_target_")
    "Node loss weights object from anemoi.training.losses."
    target_nodes: str = Field(examples=["data"])
    "name of target nodes, key in HeteroData graph object."
    node_attribute: str = Field(examples=["area_weight"])
    "name of node weight attribute, key in HeteroData graph object."


class ReweightedGraphNodeAttributeSchema(BaseModel):
    target_: Literal["anemoi.training.losses.nodeweights.ReweightedGraphNodeAttribute"] = Field(..., alias="_target_")
    "Node loss weights object from anemoi.training.losses."
    target_nodes: str = Field(examples=["data"])
    "name of target nodes, key in HeteroData graph object."
    node_attribute: str = Field(examples=["area_weight"])
    "name of node weight attribute, key in the nodes object."
    scaled_attribute: str = Field(examples=["cutout_mask"])
    "name of node attribute defining the subset of nodes to be scaled, key in the nodes object."
    weight_frac_of_total: float = Field(examples=[0.3], ge=0, le=1)
    "sum of weight of subset nodes as a fraction of sum of weight of all nodes after rescaling"


class ScaledGraphNodeAttributeSchema(BaseModel):
    target_: Literal["anemoi.training.losses.nodeweights.ScaledGraphNodeAttribute"] = Field(..., alias="_target_")
    "Node loss weights object from anemoi.training.losses."
    target_nodes: str = Field(examples=["data"])
    "name of target nodes, key in HeteroData graph object."
    node_attribute: str = Field(examples=["area_weight"])
    "name of node weight attribute, key in the nodes object."
    scaled_attribute: str = Field(examples=["dist_from_lam_center"])
    "name of node attribute defining what will the nodes be scaled with, key in the nodes object."
    inverse: bool = Field(examples=["True"])
    "Scale the weight directly proportionally to attribute or inversely."


NodeLossWeightsSchema = Union[
    GraphNodeAttributeSchema,
    ReweightedGraphNodeAttributeSchema,
    ScaledGraphNodeAttributeSchema,
]


class ScaleValidationMetrics(BaseModel):
    """Configuration for scaling validation metrics.

    Here variable scaling is possible due to the metrics being calculated in the same way as the
    training loss, within internal model space.
    """

    scalars_to_apply: list[str] = Field(example=["variable"])
    """List of scalars to be applied."""
    metrics: list[str]
    """List of metrics to keep in normalised space.."""


class BaseTrainingSchema(BaseModel):
    """Training configuration."""

    run_id: Union[str, None] = Field(example=None)
    "Run ID: used to resume a run from a checkpoint, either last.ckpt or specified in hardware.files.warm_start."
    fork_run_id: Union[str, None] = Field(example=None)
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
    num_sanity_val_steps: PositiveInt = Field(example=6)
    "Sanity check runs n batches of val before starting the training routine."
    gradient_clip: GradientClip
    "Config for gradient clipping."
    swa: SWA = Field(default_factory=SWA)
    "Config for stochastic weight averaging."
    zero_optimizer: bool = Field(example=False)
    "use ZeroRedundancyOptimizer, saves memory for larger models."
    training_loss: LossSchemas
    "Training loss configuration."
    loss_gradient_scaling: bool = False
    "Dynamic rescaling of the loss gradient. Not yet tested."
    validation_metrics: list[LossSchemas]
    "List of validation metrics configurations. These metrics "
    scale_validation_metrics: ScaleValidationMetrics
    """Configuration for scaling validation metrics."""
    rollout: Rollout = Field(default_factory=Rollout)
    "Rollout configuration."
    max_epochs: Union[PositiveInt, None] = None
    "Maximum number of epochs, stops earlier if max_steps is reached first."
    max_steps: PositiveInt = 150000
    "Maximum number of steps, stops earlier if max_epochs is reached first."
    lr: LR = Field(default_factory=LR)
    "Learning rate configuration."
    variable_loss_scaling: LossScalingSchema
    "Configuration of the variable scaling used in the loss computation."
    pressure_level_scaler: PressureLevelScalerSchema
    "Configuration of the pressure level scaler apllied in the loss computation."
    metrics: list[str]
    "List of metrics"
    node_loss_weights: NodeLossWeightsSchema
    "Node loss weights configuration."
    task: str
    "Training objective."


class ForecasterSchema(BaseTrainingSchema):
    task: str = Field(example="anemoi.training.train.forecaster.GraphForecaster")
    "Training objective."


class InterpolationSchema(BaseTrainingSchema):
    task: str = Field(example="anemoi.training.train.interpolator.GraphInterpolator")
    "Training objective."
    explicit_times: ExplicitTimes
    "Time indices for input and output."
    target_forcing: TargetForcing
    "Forcing parameters for target output times."


TrainingSchema = Union[ForecasterSchema, InterpolationSchema]
