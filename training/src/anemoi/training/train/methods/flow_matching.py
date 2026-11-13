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

    distributions = TIME_TRAINING_DISTRIBUTIONS
    name = "Flow-matching"

    def prepare(
        self,
        prepared: PreparedPredictionTarget,
    ) -> PreparedTransportObjective:
        target = prepared.model_target
        source = self.build_transport_source(prepared)
        time_level = self.draw_condition(target)

        state = target.with_sources(
            {
                name: source[name].map_with_condition(flow_matching_state, time_level[name], clean)
                for name, clean in target.items()
            },
        )
        velocity = source.zip_map_data(flow_matching_velocity, target)
        # The model target is imputed so it can be fed through the network; put the NaNs back into
        # the loss target at missing values so the loss (with ignore_nans) skips them.
        velocity = velocity.zip_map_data(
            lambda values, target_values: values.masked_fill(target_values.isnan(), float("nan")),
            prepared.model_target_missing,
        )
        return PreparedTransportObjective(
            conditioned_target=state,
            condition=time_level,
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
        state = objective.conditioned_target
        time_level = objective.condition

        def clean(velocity: torch.Tensor, state_data: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            return flow_matching_clean(state_data, velocity, time)

        return prediction.with_sources(
            {
                name: velocity.map_with_condition(clean, time_level[name], state[name])
                for name, velocity in prediction.items()
            },
        )
