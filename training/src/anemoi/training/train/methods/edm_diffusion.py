# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import TYPE_CHECKING

from anemoi.models.transport.paths import edm_loss_weight
from anemoi.models.transport.schedules import SIGMA_TRAINING_DISTRIBUTIONS
from anemoi.training.train.methods.transport_base import PreparedPredictionTarget
from anemoi.training.train.methods.transport_base import PreparedTransportObjective
from anemoi.training.train.methods.transport_base import TransportObjective
from anemoi.training.utils.index_space import IndexSpace

if TYPE_CHECKING:
    import torch

    from anemoi.models.data import Batch
    from anemoi.models.data.sources.base import Source


class EDMDiffusionTransportObjective(TransportObjective):
    """EDM diffusion objective."""

    def prepare(
        self,
        prepared: PreparedPredictionTarget,
    ) -> PreparedTransportObjective:
        model_target = prepared.model_target
        sigma = self._sample_training_sigma(
            shape={name: source.condition_shape for name, source in model_target.items()},
            device=model_target.device,
        )
        noise_weights = self._loss_weights(sigma, model_target)
        source = self.build_transport_source(prepared)
        target_noised = self._noise_target(model_target, sigma, source)
        # EDM diffusion predicts the clean target. The prediction mode decides
        # whether that clean target is a full state or a tendency field.
        # state uses DATA_FULL, tendency uses DATA_OUTPUT.
        return PreparedTransportObjective(
            conditioned_target=target_noised,
            condition=sigma,
            loss_target=prepared.loss_target,
            loss_target_layout=prepared.loss_target_layout,
            pred_layout=IndexSpace.MODEL_OUTPUT,
            weights=noise_weights,
            aux={},
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

    def compute_loss(
        self,
        y_pred: Source,
        y: Source,
        grid_shard_slice: slice | None = None,
        dataset_name: str | None = None,
        pred_layout: IndexSpace | str | None = None,
        target_layout: IndexSpace | str | None = None,
        weights: dict[str, torch.Tensor | list[torch.Tensor]] | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Compute EDM diffusion loss with noise weighting."""
        assert weights is not None, f"{self.__class__.__name__} requires weights for EDM diffusion loss computation."

        loss = self.module.loss[dataset_name]
        loss_kwargs = {
            "grid_shard_slice": grid_shard_slice,
            "group": self.module.model_comm_group,
        }
        if y_pred.is_tabular:
            loss_kwargs["per_sample_kwargs"] = {"weights": weights[dataset_name]}
        else:
            loss_kwargs["weights"] = weights[dataset_name]
        if pred_layout is not None:
            loss_kwargs["pred_layout"] = pred_layout
        if target_layout is not None:
            loss_kwargs["target_layout"] = target_layout
        if getattr(loss, "needs_shard_layout_info", False):
            loss_kwargs.update(
                grid_dim=self.module.grid_dim,
                grid_shard_sizes=self.module._grid_shard_sizes(y),
            )

        return self.module._evaluate_loss(
            loss,
            y_pred,
            y,
            gradient_scaling=self.module.loss_gradient_scaling,
            **loss_kwargs,
        )

    def _noise_target(
        self,
        x: Batch,
        sigma: dict[str, torch.Tensor],
        source: Batch,
    ) -> Batch:
        """Create the corrupted target by adding scaled source noise to the clean target."""
        return x.with_sources(
            {
                name: target.map_with_condition(
                    lambda clean, noise, sigma_sample: clean + noise * sigma_sample,
                    sigma[name],
                    source[name],
                )
                for name, target in x.items()
            },
        )

    def _sample_training_sigma(
        self,
        shape: dict[str, tuple[int, ...]],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Sample one EDM noise level per sample and ensemble member."""
        training_condition_config = dict(self.module.model.model.training_condition)
        try:
            distribution_name = training_condition_config.pop("distribution")
        except KeyError as exc:
            msg = "EDM training_condition must define 'distribution'."
            raise ValueError(msg) from exc
        if distribution_name not in SIGMA_TRAINING_DISTRIBUTIONS:
            msg = f"Unknown EDM training condition distribution: {distribution_name}"
            raise ValueError(msg)
        distribution_cls = SIGMA_TRAINING_DISTRIBUTIONS[distribution_name]
        distribution = distribution_cls(**training_condition_config)
        return distribution.sample(shape, device=device)

    def _loss_weights(
        self,
        sigma: dict[str, torch.Tensor],
        target: Batch,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Return EDM loss weights for sampled sigma values, laid out like each target's payload."""
        sigma_data = self.module.model.model.edm.sigma_data
        return {
            dataset_name: target[dataset_name].condition_per_sample(edm_loss_weight(sigma_dataset, sigma_data))
            for dataset_name, sigma_dataset in sigma.items()
        }
