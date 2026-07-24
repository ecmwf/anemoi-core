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

from anemoi.models.transport.data_helpers import Data
from anemoi.models.transport.data_helpers import add_data
from anemoi.models.transport.data_helpers import condition_shapes
from anemoi.models.transport.data_helpers import first_data_device
from anemoi.models.transport.data_helpers import multiply_batch_scalar_data
from anemoi.models.transport.data_helpers import randn_like_data
from anemoi.models.transport.data_helpers import zip_map_batch_scalar_data
from anemoi.models.transport.data_helpers import zip_map_data
from anemoi.models.transport.paths import stochastic_interpolant_alpha
from anemoi.models.transport.paths import stochastic_interpolant_alpha_dot
from anemoi.models.transport.paths import stochastic_interpolant_beta
from anemoi.models.transport.paths import stochastic_interpolant_beta_dot
from anemoi.models.transport.paths import stochastic_interpolant_bridge_noise_velocity_ratio
from anemoi.models.transport.paths import stochastic_interpolant_clean_mean
from anemoi.models.transport.paths import stochastic_interpolant_sigma
from anemoi.models.transport.schedules import TIME_TRAINING_DISTRIBUTIONS
from anemoi.training.train.methods.transport_base import PreparedPredictionTarget
from anemoi.training.train.methods.transport_base import PreparedTransportObjective
from anemoi.training.train.methods.transport_base import TransportObjective
from anemoi.training.utils.index_space import IndexSpace

if TYPE_CHECKING:
    import torch

    from anemoi.models.data import Batch


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
        # model_target is imputed so it can be fed through the network; re-mask
        # the drift loss target with NaNs at missing observations so the loss
        # (with ignore_nans) excludes them instead of fitting imputed values.
        missing = prepared.aux.get("model_target_missing")
        if missing is not None:
            drift_target = {
                name: zip_map_data(drift, missing.data[name], lambda d, m: d + m * 0)
                for name, drift in drift_target.items()
            }
        return PreparedTransportObjective(
            conditioned_target=prepared.model_target.with_data(interpolant_state),
            condition=time_level,
            loss_target=prepared.model_target.with_data(drift_target),
            loss_target_layout=IndexSpace.MODEL_OUTPUT,
            pred_layout=IndexSpace.MODEL_OUTPUT,
            weights=None,
            aux={
                "source": source,
                "interpolant_state": prepared.model_target.with_data(interpolant_state),
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
        return prediction.with_data(
            self._reconstruct_clean(
                objective.aux["interpolant_state"],
                prediction,
                objective.aux["source"],
                objective.aux["time_level"],
            ),
        )

    def _build_training_pair(
        self,
        source: dict[str, Data],
        clean_target: Batch,
    ) -> tuple[dict[str, Data], dict[str, Data], dict[str, torch.Tensor]]:
        """Create the interpolated training input and the change the model should predict."""
        target_data = clean_target.data
        time_level = self._sample_training_time(
            condition_shapes(target_data),
            device=first_data_device(target_data),
        )

        interpolant_state: dict[str, Data] = {}
        drift_target: dict[str, Data] = {}
        noise_scale = self._noise_scale
        for dataset_name, clean_dataset in target_data.items():
            time_dataset = time_level[dataset_name]
            alpha = stochastic_interpolant_alpha(time_dataset, self._alpha_schedule)
            beta = stochastic_interpolant_beta(time_dataset, self._beta_schedule)
            alpha_dot = stochastic_interpolant_alpha_dot(time_dataset, self._alpha_schedule)
            beta_dot = stochastic_interpolant_beta_dot(time_dataset, self._beta_schedule)

            source_dataset = source[dataset_name]
            interpolant_dataset = add_data(
                multiply_batch_scalar_data(source_dataset, alpha),
                multiply_batch_scalar_data(clean_dataset, beta),
            )
            drift_dataset = add_data(
                multiply_batch_scalar_data(source_dataset, alpha_dot),
                multiply_batch_scalar_data(clean_dataset, beta_dot),
            )

            if noise_scale != 0.0:
                noise = randn_like_data(
                    clean_dataset,
                    model_comm_group=getattr(self.module, "model_comm_group", None),
                    grid_shard_sizes=self.module._grid_shard_sizes(clean_target[dataset_name]),
                )
                sigma = stochastic_interpolant_sigma(
                    time_dataset,
                    schedule=self._sigma_schedule,
                    noise_scale=noise_scale,
                )
                bridge_noise = multiply_batch_scalar_data(noise, sigma)
                ratio = stochastic_interpolant_bridge_noise_velocity_ratio(
                    time_dataset,
                    schedule=self._sigma_schedule,
                    eps=1e-8,
                )
                drift_noise = multiply_batch_scalar_data(bridge_noise, ratio)
                interpolant_dataset = add_data(interpolant_dataset, bridge_noise)
                drift_dataset = add_data(drift_dataset, drift_noise)

            interpolant_state[dataset_name] = interpolant_dataset
            drift_target[dataset_name] = drift_dataset

        return interpolant_state, drift_target, time_level

    def _reconstruct_clean(
        self,
        interpolant_state: Batch,
        drift_prediction: Batch,
        source: dict[str, Data],
        time_level: dict[str, torch.Tensor],
    ) -> dict[str, Data]:
        """Estimate the clean target from the model prediction for validation metrics."""
        return {
            dataset_name: zip_map_batch_scalar_data(
                drift_prediction.data[dataset_name],
                interpolant_state.data[dataset_name],
                source[dataset_name],
                scalar=time_level[dataset_name],
                fn=lambda drift, interpolant, anchor, time: stochastic_interpolant_clean_mean(
                    drift=drift,
                    interpolant=interpolant,
                    anchor=anchor,
                    t=time,
                    alpha_schedule=self._alpha_schedule,
                    beta_schedule=self._beta_schedule,
                    sigma_schedule=self._sigma_schedule,
                    noise_scale=self._noise_scale,
                ),
            )
            for dataset_name in interpolant_state.data
        }

    def _sample_training_time(
        self,
        shape: dict[str, tuple[int, ...]],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Draw one interpolation time per sample and ensemble member."""
        training_condition_config = dict(self.module.model.model.training_condition)
        try:
            distribution_name = training_condition_config.pop("distribution")
        except KeyError as exc:
            msg = "Stochastic-interpolant training_condition must define 'distribution'."
            raise ValueError(msg) from exc
        if distribution_name not in TIME_TRAINING_DISTRIBUTIONS:
            msg = f"Unknown stochastic-interpolant training condition distribution: {distribution_name}"
            raise ValueError(msg)
        distribution_cls = TIME_TRAINING_DISTRIBUTIONS[distribution_name]
        distribution = distribution_cls(**training_condition_config)
        return distribution.sample(shape, device=device)

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
