# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import torch

from anemoi.models.transport.paths import stochastic_interpolant_alpha
from anemoi.models.transport.paths import stochastic_interpolant_alpha_dot
from anemoi.models.transport.paths import stochastic_interpolant_beta
from anemoi.models.transport.paths import stochastic_interpolant_beta_dot
from anemoi.models.transport.paths import stochastic_interpolant_bridge_noise_velocity_ratio
from anemoi.models.transport.paths import stochastic_interpolant_clean_mean
from anemoi.models.transport.paths import stochastic_interpolant_sigma
from anemoi.models.transport.random_fields import randn_like_with_grid_sharding
from anemoi.training.train.methods.transport_base import PreparedPredictionTarget
from anemoi.training.train.methods.transport_base import PreparedTransportObjective
from anemoi.training.train.methods.transport_base import TransportObjective
from anemoi.training.utils.index_space import IndexSpace


class StochasticInterpolantTransportObjective(TransportObjective):
    """Stochastic-interpolant objective between a source field and the target field."""

    def prepare(
        self,
        prepared: PreparedPredictionTarget,
    ) -> PreparedTransportObjective:
        source = self.build_transport_source(prepared)

        interpolant_state, drift_target, time_level = self._build_training_pair(
            source,
            prepared.model_target,
        )
        return PreparedTransportObjective(
            conditioned_target=interpolant_state,
            condition=time_level,
            loss_target=drift_target,
            loss_target_layout=IndexSpace.MODEL_OUTPUT,
            pred_layout=IndexSpace.MODEL_OUTPUT,
            weights=None,
            aux={
                "source": source,
                "interpolant_state": interpolant_state,
                "time_level": time_level,
            },
        )

    def forward(
        self,
        x: dict[str, torch.Tensor],
        conditioned_target: dict[str, torch.Tensor],
        condition: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return self.module.model.model(
            x,
            conditioned_target,
            condition,
            model_comm_group=self.module.model_comm_group,
            grid_shard_shapes=self.module.grid_shard_shapes,
        )

    def reconstruct_endpoint(
        self,
        prediction: dict[str, torch.Tensor],
        objective: PreparedTransportObjective,
    ) -> dict[str, torch.Tensor]:
        return self._reconstruct_clean(
            objective.aux["interpolant_state"],
            prediction,
            objective.aux["source"],
            objective.aux["time_level"],
        )

    def _build_training_pair(
        self,
        source: dict[str, torch.Tensor],
        clean_target: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Create the interpolated training input and the change the model should predict."""
        shape = {dataset_name: target.shape for dataset_name, target in clean_target.items()}
        time_level = self._sample_time_level(shape, next(iter(clean_target.values())).device)

        interpolant_state: dict[str, torch.Tensor] = {}
        drift_target: dict[str, torch.Tensor] = {}
        noise_scale = self._noise_scale
        for dataset_name, clean_dataset in clean_target.items():
            time_dataset = time_level[dataset_name]
            alpha = stochastic_interpolant_alpha(time_dataset, self._alpha_schedule)
            beta = stochastic_interpolant_beta(time_dataset, self._beta_schedule)
            alpha_dot = stochastic_interpolant_alpha_dot(time_dataset, self._alpha_schedule)
            beta_dot = stochastic_interpolant_beta_dot(time_dataset, self._beta_schedule)
            bridge_noise = 0.0
            drift_noise = 0.0
            if noise_scale != 0.0:
                grid_shard_shapes = getattr(self.module, "grid_shard_shapes", None)
                noise = randn_like_with_grid_sharding(
                    clean_dataset,
                    model_comm_group=getattr(self.module, "model_comm_group", None),
                    grid_shard_shapes=grid_shard_shapes.get(dataset_name) if grid_shard_shapes is not None else None,
                )
                sigma = stochastic_interpolant_sigma(
                    time_dataset,
                    schedule=self._sigma_schedule,
                    noise_scale=noise_scale,
                )
                bridge_noise = sigma * noise
                ratio = stochastic_interpolant_bridge_noise_velocity_ratio(
                    time_dataset,
                    schedule=self._sigma_schedule,
                    eps=1e-8,
                )
                drift_noise = ratio * bridge_noise
            interpolant_state[dataset_name] = alpha * source[dataset_name] + beta * clean_dataset + bridge_noise
            drift_target[dataset_name] = alpha_dot * source[dataset_name] + beta_dot * clean_dataset + drift_noise

        return interpolant_state, drift_target, time_level

    def _reconstruct_clean(
        self,
        interpolant_state: dict[str, torch.Tensor],
        drift_prediction: dict[str, torch.Tensor],
        source: dict[str, torch.Tensor],
        time_level: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Estimate the clean target from the model prediction for validation metrics."""
        return {
            dataset_name: stochastic_interpolant_clean_mean(
                drift=drift_prediction[dataset_name],
                interpolant=interpolant_state[dataset_name],
                anchor=source[dataset_name],
                t=time_level[dataset_name],
                alpha_schedule=self._alpha_schedule,
                beta_schedule=self._beta_schedule,
                sigma_schedule=self._sigma_schedule,
                noise_scale=self._noise_scale,
            )
            for dataset_name in interpolant_state
        }

    def _sample_time_level(
        self,
        shape: dict[str, tuple[int, ...]],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Draw one interpolation time per sample and ensemble member."""
        dataset_names = list(shape.keys())
        ref_shape = shape[dataset_names[0]]
        assert (
            len(ref_shape) == 5
        ), "Expected 5D tensor shape (batch, time, ensemble, grid, vars) for stochastic interpolants."
        batch_size = ref_shape[0]
        ensemble_size = ref_shape[2]
        for dataset_name, shape_x in shape.items():
            assert len(shape_x) == 5, f"Expected 5D tensor shape for dataset '{dataset_name}'."
            assert (
                shape_x[0] == batch_size and shape_x[2] == ensemble_size
            ), "Batch or ensemble dimension mismatch across datasets when sampling stochastic-interpolant times."

        eps = 1e-7
        time_base = eps + (1.0 - 2.0 * eps) * torch.rand((batch_size, ensemble_size), device=device)
        time_base = time_base[:, None, :, None, None]
        # Important: the model later reads the condition from one dataset and
        # assumes every dataset carries the same bridge time. Keep this shared
        # across datasets unless the model conditioning path is changed too.
        return {dataset_name: time_base.expand(shape_x) for dataset_name, shape_x in shape.items()}

    @property
    def _alpha_schedule(self) -> str:
        return self.module.model.model.stochastic_interpolant.alpha_schedule

    @property
    def _beta_schedule(self) -> str:
        return self.module.model.model.stochastic_interpolant.beta_schedule

    @property
    def _sigma_schedule(self) -> str:
        return self.module.model.model.stochastic_interpolant.sigma_schedule

    @property
    def _noise_scale(self) -> float:
        return self.module.model.model.stochastic_interpolant.noise_scale
