# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import TYPE_CHECKING

from anemoi.models.transport.paths import flow_matching_clean
from anemoi.models.transport.paths import flow_matching_state
from anemoi.models.transport.paths import flow_matching_velocity
from anemoi.models.transport.schedules import TIME_TRAINING_DISTRIBUTIONS
from anemoi.training.train.methods.transport_base import PreparedPredictionTarget
from anemoi.training.train.methods.transport_base import PreparedTransportObjective
from anemoi.training.train.methods.transport_base import TransportObjective
from anemoi.training.utils.index_space import IndexSpace

if TYPE_CHECKING:
    import torch

    from anemoi.models.data import Batch


class FlowMatchingTransportObjective(TransportObjective):
    """Flow-matching objective: learn the velocity of the straight path from Gaussian noise to the target."""

    def prepare(
        self,
        prepared: PreparedPredictionTarget,
    ) -> PreparedTransportObjective:
        target = prepared.model_target
        source = self.build_transport_source(prepared)
        time_level = self._sample_training_time(
            {name: clean.condition_shape for name, clean in target.items()},
            device=target.device,
        )

        state = target.with_sources(
            {
                name: source[name].map_with_condition(flow_matching_state, time_level[name], clean)
                for name, clean in target.items()
            },
        )
        velocity = source.zip_map_data(flow_matching_velocity, target)
        # The model target is imputed so it can be fed through the network; put the NaNs back into
        # the loss target at missing values so the loss (with ignore_nans) skips them.
        missing = prepared.aux.get("model_target_missing")
        if missing is not None:
            velocity = velocity.zip_map_data(
                lambda values, target_values: values.masked_fill(target_values.isnan(), float("nan")),
                missing,
            )
        return PreparedTransportObjective(
            conditioned_target=state,
            condition=time_level,
            loss_target=velocity,
            loss_target_layout=IndexSpace.MODEL_OUTPUT,
            pred_layout=IndexSpace.MODEL_OUTPUT,
            weights=None,
            aux={
                "state": state,
                "time_level": time_level,
            },
        )

    def forward(
        self,
        x: Batch,
        conditioned_target: Batch,
        condition: dict[str, torch.Tensor],
        target_forcing: Batch | None = None,
    ) -> Batch:
        return self.module.model.model(
            x,
            conditioned_target,
            condition,
            model_comm_group=self.module.model_comm_group,
            target_forcing=target_forcing,
        )

    def reconstruct_endpoint(
        self,
        prediction: Batch,
        objective: PreparedTransportObjective,
    ) -> Batch:
        """Estimate the clean target from the predicted velocity, for validation metrics."""
        state = objective.aux["state"]
        time_level = objective.aux["time_level"]

        def clean(velocity: torch.Tensor, state_data: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            return flow_matching_clean(state_data, velocity, time)

        return prediction.with_sources(
            {
                name: velocity.map_with_condition(clean, time_level[name], state[name])
                for name, velocity in prediction.items()
            },
        )

    def _sample_training_time(
        self,
        shape: dict[str, tuple[int, ...]],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Draw one time per sample and ensemble member."""
        training_condition_config = dict(self.module.model.model.training_condition)
        try:
            distribution_name = training_condition_config.pop("distribution")
        except KeyError as exc:
            msg = "Flow-matching training_condition must define 'distribution'."
            raise ValueError(msg) from exc
        if distribution_name not in TIME_TRAINING_DISTRIBUTIONS:
            msg = f"Unknown flow-matching training condition distribution: {distribution_name}"
            raise ValueError(msg)
        distribution_cls = TIME_TRAINING_DISTRIBUTIONS[distribution_name]
        distribution = distribution_cls(**training_condition_config)
        return distribution.sample(shape, device=device)
