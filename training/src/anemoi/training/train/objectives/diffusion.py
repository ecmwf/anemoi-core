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

from .base import ObjectiveStrategy
from .base import validate_schedule_shapes


class DiffusionObjective(ObjectiveStrategy):
    """Diffusion objective (EDM-style preconditioning)."""

    def __init__(self, rho: float = 7.0) -> None:
        self.rho = rho

    def sample_schedule(
        self,
        shape: dict[str, tuple[int, ...]],
        device: torch.device,
        *,
        model: torch.nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        # Sample a noise scale per batch element for diffusion training.
        assert model is not None, "DiffusionObjective requires a model to provide sigma_min/max."

        batch_size, ensemble_size = validate_schedule_shapes(shape)

        sigma_max = model.sigma_max
        sigma_min = model.sigma_min

        base_shape = (batch_size, ensemble_size)
        rnd_uniform = torch.rand(base_shape, device=device)
        sigma_base = (
            sigma_max ** (1.0 / self.rho)
            + rnd_uniform * (sigma_min ** (1.0 / self.rho) - sigma_max ** (1.0 / self.rho))
        ) ** self.rho
        sigma_base = sigma_base[:, None, :, None, None]

        return dict.fromkeys(shape, sigma_base)

    def build_training_pair(
        self,
        y: dict[str, torch.Tensor],
        schedule: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # Conditioning is a noised state; target is the clean state.
        y_noised = {name: y[name] + torch.randn_like(y[name]) * schedule[name] for name in y}
        return y_noised, y

    def forward(
        self,
        model: torch.nn.Module,
        x: dict[str, torch.Tensor],
        y_cond: dict[str, torch.Tensor],
        schedule: dict[str, torch.Tensor],
        *,
        model_comm_group: object | None = None,
        grid_shard_shapes: dict[str, tuple[int, ...] | None] | None = None,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        # Model predicts denoised output using EDM preconditioning.
        return model.fwd_with_preconditioning(
            x,
            y_cond,
            schedule,
            model_comm_group=model_comm_group,
            grid_shard_shapes=grid_shard_shapes,
        )

    def pre_loss_weights(
        self,
        schedule: dict[str, torch.Tensor],
        *,
        model: torch.nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        # Scale loss by noise level (EDM weighting).
        assert model is not None, "DiffusionObjective requires a model to provide sigma_data."

        sigma_data = model.sigma_data
        dataset_names = list(schedule)
        sigma_ref = schedule[dataset_names[0]]
        sigma_base = sigma_ref[:, 0, :, 0, 0]
        weight_base = (sigma_base**2 + sigma_data**2) / (sigma_base * sigma_data) ** 2
        weight_base = weight_base[:, None, :, None, None]

        return dict.fromkeys(schedule, weight_base)
