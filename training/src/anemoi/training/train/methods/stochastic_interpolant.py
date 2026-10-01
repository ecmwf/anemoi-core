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
    from anemoi.models.data.sources import Source


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
            drift_target = drift_target.zip_map_data(
                lambda values, target: values.masked_fill(target.isnan(), float("nan")),
                missing,
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
        return self._reconstruct_clean(
            objective.aux["interpolant_state"],
            prediction,
            objective.aux["source"],
            objective.aux["time_level"],
        )

    def _build_training_pair(
        self,
        source: Batch,
        clean_target: Batch,
    ) -> tuple[Batch, Batch, dict[str, torch.Tensor]]:
        """Create the interpolated training input and the change the model should predict."""
        time_level = self._sample_training_time(
            {name: clean.condition_shape for name, clean in clean_target.items()},
            device=clean_target.device,
        )

        def scaled(values: Source, factor: torch.Tensor) -> Source:
            return values.map_with_condition(lambda data, sample_factor: data * sample_factor, factor)

        def added(left: Source, right: Source) -> Source:
            return left.zip_map_data(lambda left_data, right_data: left_data + right_data, right)

        interpolant_state: dict[str, Source] = {}
        drift_target: dict[str, Source] = {}
        noise_scale = self._noise_scale
        for dataset_name, clean in clean_target.items():
            time_dataset = time_level[dataset_name]
            alpha = stochastic_interpolant_alpha(time_dataset, self._alpha_schedule)
            beta = stochastic_interpolant_beta(time_dataset, self._beta_schedule)
            alpha_dot = stochastic_interpolant_alpha_dot(time_dataset, self._alpha_schedule)
            beta_dot = stochastic_interpolant_beta_dot(time_dataset, self._beta_schedule)

            anchor = source[dataset_name]
            interpolant = added(scaled(clean, beta), scaled(anchor, alpha))
            drift = added(scaled(clean, beta_dot), scaled(anchor, alpha_dot))

            if noise_scale != 0.0:
                noise = clean.randn_like(model_comm_group=getattr(self.module, "model_comm_group", None))
                sigma = stochastic_interpolant_sigma(
                    time_dataset,
                    schedule=self._sigma_schedule,
                    noise_scale=noise_scale,
                )
                bridge_noise = scaled(noise, sigma)
                ratio = stochastic_interpolant_bridge_noise_velocity_ratio(
                    time_dataset,
                    schedule=self._sigma_schedule,
                    eps=1e-8,
                )
                interpolant = added(interpolant, bridge_noise)
                drift = added(drift, scaled(bridge_noise, ratio))

            interpolant_state[dataset_name] = interpolant
            drift_target[dataset_name] = drift

        return clean_target.with_sources(interpolant_state), clean_target.with_sources(drift_target), time_level

    def _reconstruct_clean(
        self,
        interpolant_state: Batch,
        drift_prediction: Batch,
        source: Batch,
        time_level: dict[str, torch.Tensor],
    ) -> Batch:
        """Estimate the clean target from the model prediction for validation metrics."""

        def clean_mean(
            drift: torch.Tensor,
            interpolant: torch.Tensor,
            anchor: torch.Tensor,
            time: torch.Tensor,
        ) -> torch.Tensor:
            return stochastic_interpolant_clean_mean(
                drift=drift,
                interpolant=interpolant,
                anchor=anchor,
                t=time,
                alpha_schedule=self._alpha_schedule,
                beta_schedule=self._beta_schedule,
                sigma_schedule=self._sigma_schedule,
                noise_scale=self._noise_scale,
            )

        return drift_prediction.with_sources(
            {
                dataset_name: drift_prediction[dataset_name].map_with_condition(
                    clean_mean,
                    time_level[dataset_name],
                    interpolant_state[dataset_name],
                    source[dataset_name],
                )
                for dataset_name in interpolant_state
            },
        )

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
