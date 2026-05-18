# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import torch

from anemoi.models.transport.paths import edm_loss_weight
from anemoi.models.transport.paths import karras_sigma_from_unit_time
from anemoi.training.train.methods.transport_base import PreparedPredictionTarget
from anemoi.training.train.methods.transport_base import PreparedTransportObjective
from anemoi.training.train.methods.transport_base import TransportObjective
from anemoi.training.utils.index_space import IndexSpace


class EDMDiffusionTransportObjective(TransportObjective):
    """EDM diffusion objective."""

    def prepare(
        self,
        prepared: PreparedPredictionTarget,
    ) -> PreparedTransportObjective:
        shapes = {dataset_name: target.shape for dataset_name, target in prepared.model_target.items()}
        sigma, noise_weights = self._get_noise_level(
            shape=shapes,
            sigma_max=self.module.model.model.edm.sigma_max,
            sigma_min=self.module.model.model.edm.sigma_min,
            sigma_data=self.module.model.model.edm.sigma_data,
            rho=self._rho,
            device=next(iter(prepared.model_target.values())).device,
        )
        source = self.build_transport_source(prepared)
        target_noised = self._noise_target(prepared.model_target, sigma, source)
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

    @property
    def _rho(self) -> float:
        if hasattr(self.module, "rho"):
            return self.module.rho
        return self.module.config.model.model.transport.rho

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
            grid_shard_sizes=self.module.grid_shard_sizes,
        )

    def compute_loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        grid_shard_slice: slice | None = None,
        dataset_name: str | None = None,
        pred_layout: IndexSpace | str | None = None,
        target_layout: IndexSpace | str | None = None,
        weights: dict[str, torch.Tensor] | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Compute EDM diffusion loss with noise weighting."""
        assert weights is not None, f"{self.__class__.__name__} requires weights for EDM diffusion loss computation."

        loss = self.module.loss[dataset_name]
        loss_kwargs = {
            "weights": weights[dataset_name],
            "grid_shard_slice": grid_shard_slice,
            "group": self.module.model_comm_group,
        }
        if pred_layout is not None:
            loss_kwargs["pred_layout"] = pred_layout
        if target_layout is not None:
            loss_kwargs["target_layout"] = target_layout
        if getattr(loss, "needs_shard_layout_info", False):
            loss_kwargs.update(
                grid_dim=self.module.grid_dim,
                grid_shard_sizes=self.module.grid_shard_sizes[dataset_name],
            )

        return loss(y_pred, y, **loss_kwargs)

    def _noise_target(
        self,
        x: dict[str, torch.Tensor],
        sigma: dict[str, torch.Tensor],
        source: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Create the corrupted target by adding scaled source noise to the clean target."""
        return {name: x[name] + source[name] * sigma[name] for name in x}

    def _get_noise_level(
        self,
        shape: dict[str, tuple[int, ...]],
        sigma_max: float,
        sigma_min: float,
        sigma_data: float,
        rho: float,
        device: torch.device,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Sample one noise level per sample and return the matching EDM loss weight."""
        sigma, weight = {}, {}
        dataset_names = list(shape.keys())
        ref_shape = shape[dataset_names[0]]
        assert (
            len(ref_shape) == 5
        ), "Expected 5D tensor shape (batch, time, ensemble, grid, vars) for EDM diffusion noise."
        batch_size = ref_shape[0]
        ensemble_size = ref_shape[2]
        for dataset_name, shape_x in shape.items():
            assert (
                len(shape_x) == 5
            ), f"Expected 5D tensor shape (batch, time, ensemble, grid, vars) for dataset '{dataset_name}'."
            assert (
                shape_x[0] == batch_size and shape_x[2] == ensemble_size
            ), "Batch or ensemble dimension mismatch across datasets when sampling EDM diffusion noise."

        base_shape = (batch_size, ensemble_size)
        rnd_uniform = torch.rand(base_shape, device=device)
        sigma_base = karras_sigma_from_unit_time(
            rnd_uniform,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            rho=rho,
        )
        weight_base = edm_loss_weight(sigma_base, sigma_data)
        sigma_base = sigma_base[:, None, :, None, None]
        weight_base = weight_base[:, None, :, None, None]

        # Important: the model later reads the condition from one dataset and
        # assumes every dataset carries the same noise level. Keep this shared
        # across datasets unless the model conditioning path is changed too.
        for dataset_name in shape:
            sigma[dataset_name] = sigma_base
            weight[dataset_name] = weight_base
        return sigma, weight
