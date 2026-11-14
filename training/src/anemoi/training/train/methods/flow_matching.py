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
from anemoi.models.transport.paths import flow_matching_noisy_target
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

    distributions = TIME_TRAINING_DISTRIBUTIONS
    name = "Flow-matching"

    def prepare(
        self,
        prepared: PreparedPredictionTarget,
    ) -> PreparedTransportObjective:
        target = prepared.model_target
        noise = self.draw_noise(prepared)
        time = self.draw_condition(target)

        noisy_target = target.with_sources(
            {
                name: noise[name].map_with_condition(flow_matching_noisy_target, time[name], clean)
                for name, clean in target.items()
            },
        )
        velocity = noise.zip_map_data(flow_matching_velocity, target)
        # The model target is imputed so it can be fed through the network; put the NaNs back into
        # the loss target at missing values so the loss (with ignore_nans) skips them.
        velocity = velocity.zip_map_data(
            lambda values, target_values: values.masked_fill(target_values.isnan(), float("nan")),
            prepared.model_target_missing,
        )
        return PreparedTransportObjective(
            noisy_target=noisy_target,
            condition=time,
            loss_target=velocity,
            loss_target_layout=IndexSpace.MODEL_OUTPUT,
            pred_layout=IndexSpace.MODEL_OUTPUT,
            weights=None,
        )

    def reconstruct_endpoint(
        self,
        prediction: Batch,
        objective: PreparedTransportObjective,
    ) -> Batch:
        """Estimate the clean target from the predicted velocity, for validation metrics."""
        noisy_target = objective.noisy_target
        time = objective.condition

        def clean(velocity: torch.Tensor, noisy_target_data: torch.Tensor, time_data: torch.Tensor) -> torch.Tensor:
            return flow_matching_clean(noisy_target_data, velocity, time_data)

        return prediction.with_sources(
            {
                name: velocity.map_with_condition(clean, time[name], noisy_target[name])
                for name, velocity in prediction.items()
            },
        )
