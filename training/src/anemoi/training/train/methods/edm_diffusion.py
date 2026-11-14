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
from typing import Any

from anemoi.models.transport.paths import edm_loss_weight
from anemoi.models.transport.schedules import SIGMA_TRAINING_DISTRIBUTIONS
from anemoi.training.losses.base import BaseLoss
from anemoi.training.train.methods.transport_base import PreparedPredictionTarget
from anemoi.training.train.methods.transport_base import PreparedTransportObjective
from anemoi.training.train.methods.transport_base import TransportObjective
from anemoi.training.utils.index_space import IndexSpace

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch

    from anemoi.models.data import Batch
    from anemoi.models.data.sources.base import Source


class EDMDiffusionTransportObjective(TransportObjective):
    """EDM diffusion objective."""

    distributions = SIGMA_TRAINING_DISTRIBUTIONS
    name = "EDM"

    def prepare(
        self,
        prepared: PreparedPredictionTarget,
    ) -> PreparedTransportObjective:
        model_target = prepared.model_target
        sigma = self.draw_condition(model_target)
        noise_weights = self._loss_weights(sigma, model_target)
        noise = self.draw_noise(prepared)
        target_noised = self._noise_target(model_target, sigma, noise)
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
        )

    def loss_kwargs(
        self,
        y_pred: Source,
        dataset_name: str,
        weights: dict[str, torch.Tensor | list[torch.Tensor]] | None,
    ) -> dict[str, Any]:
        """Return the noise-level weights of ``dataset_name``, laid out for the loss.

        :meth:`check_losses` has made sure that the loss applies them.
        """
        assert weights is not None, f"{self.__class__.__name__} requires weights for EDM diffusion loss computation."
        dataset_weights = weights[dataset_name]
        if y_pred.is_tabular:
            return {"per_sample_kwargs": {"weights": dataset_weights}}
        return {"weights": dataset_weights}

    def check_losses(self, losses: Mapping[str, BaseLoss]) -> None:
        """Reject a training loss that would leave the EDM noise-level weights unused.

        Such a loss would train a different objective without any warning.
        """
        for dataset_name, loss in losses.items():
            leaves = loss.iter_leaf_losses() if isinstance(loss, BaseLoss) else [loss]
            ignoring = sorted({type(leaf).__name__ for leaf in leaves if not getattr(leaf, "applies_weights", False)})
            if ignoring:
                msg = (
                    f"EDM diffusion weights each sample's loss by its noise level, but the training loss of dataset "
                    f"'{dataset_name}' leaves the weights unused ({', '.join(ignoring)}). Use a loss that applies "
                    "them, such as anemoi.training.losses.MSELoss, as in training/edm_diffusion.yaml."
                )
                raise ValueError(msg)

    def _noise_target(
        self,
        x: Batch,
        sigma: dict[str, torch.Tensor],
        noise: Batch,
    ) -> Batch:
        """Create the corrupted target by adding noise, scaled by sigma, to the clean target."""
        return x.with_sources(
            {
                name: target.map_with_condition(
                    lambda clean, noise, sigma_sample: clean + noise * sigma_sample,
                    sigma[name],
                    noise[name],
                )
                for name, target in x.items()
            },
        )

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
