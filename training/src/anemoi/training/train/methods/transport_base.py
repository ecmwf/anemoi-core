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
from typing import TYPE_CHECKING
from typing import Any

from anemoi.models.transport import TransportSourceRequest
from anemoi.training.train.methods.base import BaseTrainingModule

if TYPE_CHECKING:
    import torch

    from anemoi.training.train.methods.transport import TransportTraining
    from anemoi.training.utils.index_space import IndexSpace


@dataclass
class PreparedPredictionTarget:
    """Targets prepared by the selected prediction mode.

    The prediction mode decides whether transport is trained on future states
    or on tendencies. It also keeps the targets needed later for the loss and
    validation metrics, because those may live in different variable spaces.

    model_target:
        Target in MODEL_OUTPUT space. This is the tensor that diffusion or SI
        corrupts before passing it to the model.

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

    aux:
        Extra prediction-mode data needed later, such as the latest input state
        for tendency reconstruction or a reference-state source for transport.
    """

    model_target: dict[str, torch.Tensor]
    loss_target: dict[str, torch.Tensor]
    loss_target_layout: IndexSpace
    metric_target: dict[str, torch.Tensor]
    aux: dict[str, Any]


@dataclass
class PreparedTransportObjective:
    """Inputs and targets prepared by the selected transport objective.

    The transport objective decides how to corrupt the prediction target and
    what the model should learn from that corrupted input. Diffusion predicts a
    clean endpoint. SI predicts the bridge drift.

    conditioned_target:
        Corrupted target passed to the model together with the normal input.
        For diffusion this is the noised target. For SI this is the interpolated
        bridge state.

    condition:
        Scalar conditioning value expanded to tensor form. For diffusion this is
        sigma. For SI this is the bridge time.

    loss_target:
        Target used by the loss after the transport objective has prepared it.
        For diffusion this is usually the clean endpoint. For SI this is the
        drift target.

    loss_target_layout:
        Variable layout of loss_target. The loss uses this to align loss_target
        with the model prediction.

    pred_layout:
        Variable layout of the model prediction passed to the loss. Transport
        models currently predict in MODEL_OUTPUT space.

    weights:
        Optional per-sample/per-variable loss weights. Diffusion uses this for
        EDM noise-level weighting. SI usually leaves this as None.

    aux:
        Extra objective-specific tensors needed later, for example SI source,
        bridge state, and bridge time for validation reconstruction.
    """

    conditioned_target: dict[str, torch.Tensor]
    condition: dict[str, torch.Tensor]
    loss_target: dict[str, torch.Tensor]
    loss_target_layout: IndexSpace
    pred_layout: IndexSpace
    weights: dict[str, torch.Tensor] | None
    aux: dict[str, Any]


class TransportObjective:
    """Common interface for training objectives such as diffusion and stochastic interpolants."""

    def __init__(self, module: TransportTraining) -> None:
        self.module = module

    def prepare(
        self,
        prepared: PreparedPredictionTarget,
    ) -> PreparedTransportObjective:
        raise NotImplementedError

    def forward(
        self,
        x: dict[str, torch.Tensor],
        conditioned_target: dict[str, torch.Tensor],
        condition: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def reconstruct_endpoint(
        self,
        prediction: dict[str, torch.Tensor],
        objective: PreparedTransportObjective,
    ) -> dict[str, torch.Tensor]:
        del objective
        return prediction

    def prepare_loss_prediction(
        self,
        prediction: dict[str, torch.Tensor],
        objective: PreparedTransportObjective,
    ) -> dict[str, torch.Tensor]:
        del objective
        return prediction

    def compute_loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        grid_shard_slice: slice | None = None,
        dataset_name: str | None = None,
        pred_layout: IndexSpace | str | None = None,
        target_layout: IndexSpace | str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return BaseTrainingModule._compute_loss(
            self.module,
            y_pred=y_pred,
            y=y,
            grid_shard_slice=grid_shard_slice,
            dataset_name=dataset_name,
            pred_layout=pred_layout,
            target_layout=target_layout,
            **kwargs,
        )

    def build_transport_source(
        self,
        prepared: PreparedPredictionTarget,
        default_kind: str = "gaussian",
    ) -> dict[str, torch.Tensor]:
        def reference_source_factory() -> dict[str, torch.Tensor]:
            reference = prepared.aux.get("transport_reference_source")
            if reference is None:
                msg = "Transport source kind 'reference_state' requires a reference source in the prediction mode."
                raise ValueError(msg)
            if callable(reference):
                # Materialise lazily built references only when reference_state is selected.
                return reference()
            return reference

        request = TransportSourceRequest.from_tensors(
            prepared.model_target,
            default_kind=default_kind,
            source_factories={"reference_state": reference_source_factory},
            model_comm_group=getattr(self.module, "model_comm_group", None),
            grid_shard_sizes=getattr(self.module, "grid_shard_sizes", None),
            error_context="training",
        )
        return self.module.model.model.transport_source.build(request)
