# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from anemoi.models.transport.registry import build_registered

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch

    from anemoi.models.data import Batch
    from anemoi.models.data.sources import Source
    from anemoi.training.losses.base import BaseLoss
    from anemoi.training.train.methods.transport import TransportTraining
    from anemoi.training.utils.index_space import IndexSpace


@dataclass
class PreparedPredictionTarget:
    """Targets prepared by the selected prediction mode.

    The prediction mode decides whether transport is trained on future states
    or on tendencies. It also keeps the targets needed later for the loss and
    validation metrics, because those may live in different variable spaces.

    model_target:
        Target in MODEL_OUTPUT space. This is the tensor that EDM diffusion or flow
        matching corrupts before passing it to the model.

    loss_target:
        Clean target used by the training loss before the transport objective
        changes it. For state prediction this is usually DATA_FULL. For tendency
        prediction this is usually DATA_OUTPUT.

    loss_target_layout:
        Variable layout of loss_target. The loss uses this to align loss_target
        with the model prediction.

    metric_target:
        Clean state target used for validation metrics. For tendency prediction,
        the model output is reconstructed back to state space before comparing
        against this target.

    model_target_missing:
        Counterpart of model_target that keeps the NaNs of missing values.
        Objectives that derive their loss target from model_target (the
        flow-matching velocity) use it to mask the missing values again.

    target_forcing:
        Forcings at the output times, normalised like the model inputs, which
        the decoder reads.

    reference:
        Latest input state that predicted tendencies are added to. Set by the
        tendency prediction mode only.
    """

    model_target: Batch
    loss_target: Batch
    loss_target_layout: IndexSpace
    metric_target: Batch
    model_target_missing: Batch
    target_forcing: Batch | None = None
    reference: Batch | None = None


@dataclass
class PreparedTransportObjective:
    """Inputs and targets prepared by the selected transport objective.

    The transport objective decides how to corrupt the prediction target and
    what the model should learn from that corrupted input. EDM diffusion
    predicts a clean endpoint. Flow matching predicts a velocity.

    conditioned_target:
        Corrupted target passed to the model together with the normal input.
        For EDM diffusion this is the noised target. For flow matching this is the
        point on the path from the source field to the target.

    condition:
        Scalar conditioning value expanded to tensor form. For EDM diffusion this is
        sigma. For flow matching this is the time.

    loss_target:
        Target used by the loss after the transport objective has prepared it.
        For EDM diffusion this is usually the clean endpoint. For flow matching this
        is the velocity.

    loss_target_layout:
        Variable layout of loss_target. The loss uses this to align loss_target
        with the model prediction.

    pred_layout:
        Variable layout of the model prediction passed to the loss. Transport
        models currently predict in MODEL_OUTPUT space.

    weights:
        Optional per-sample/per-variable loss weights. EDM diffusion uses this for
        EDM noise-level weighting. Flow matching leaves this as None.
    """

    conditioned_target: Batch
    condition: dict[str, torch.Tensor]
    loss_target: Batch
    loss_target_layout: IndexSpace
    pred_layout: IndexSpace
    weights: dict[str, torch.Tensor | list[torch.Tensor]] | None


class TransportObjective:
    """Common interface for training objectives such as EDM diffusion and flow matching.

    Subclasses name the registry of training distributions their condition is drawn from.
    """

    distributions: ClassVar[dict[str, type]]
    name: ClassVar[str]

    def __init__(self, module: TransportTraining) -> None:
        self.module = module

    @cached_property
    def distribution(self) -> Any:
        """The training distribution of the noise level or time, built the first time it is needed."""
        settings = dict(self.module.model.model.training_condition)
        try:
            distribution_name = settings.pop("distribution")
        except KeyError as exc:
            msg = f"{self.name} training_condition must define 'distribution'."
            raise ValueError(msg) from exc
        return build_registered(
            self.distributions,
            distribution_name,
            settings,
            f"{self.name} training condition distribution",
        )

    def draw_condition(self, target: Batch) -> dict[str, torch.Tensor]:
        """Draw one noise level or time per sample and ensemble member, shaped for each dataset of ``target``."""
        return self.distribution.sample(
            {name: source.condition_shape for name, source in target.items()},
            device=target.device,
        )

    def check_losses(self, losses: Mapping[str, BaseLoss]) -> None:
        """Check, once when training is set up, that the training losses suit this objective."""
        del losses

    def prepare(
        self,
        prepared: PreparedPredictionTarget,
    ) -> PreparedTransportObjective:
        raise NotImplementedError

    def reconstruct_endpoint(
        self,
        prediction: Batch,
        objective: PreparedTransportObjective,
    ) -> Batch:
        del objective
        return prediction

    def prepare_loss_prediction(
        self,
        prediction: Batch,
        objective: PreparedTransportObjective,
    ) -> Batch:
        del objective
        return prediction

    def loss_kwargs(
        self,
        y_pred: Source,
        dataset_name: str,
        weights: dict[str, torch.Tensor | list[torch.Tensor]] | None,
    ) -> dict[str, Any]:
        """Return further arguments for the training loss of ``dataset_name``, such as per-sample weights."""
        del y_pred, dataset_name, weights
        return {}

    def draw_noise(self, prepared: PreparedPredictionTarget) -> Batch:
        """Draw the Gaussian noise the transport starts from, shaped and described like the model target."""
        # The model target also carries the grid shard sizes the noise is split by.
        group = getattr(self.module, "model_comm_group", None)
        return prepared.model_target.with_sources(
            {name: target.randn_like(group) for name, target in prepared.model_target.items()},
        )
