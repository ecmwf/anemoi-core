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


class FlowObjective(ObjectiveStrategy):
    """Flow-matching objective with linear interpolation path."""

    def sample_schedule(
        self,
        shape: dict[str, tuple[int, ...]],
        device: torch.device,
        *,
        model: torch.nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        # Sample interpolation time t in [0, 1].
        del model
        batch_size, ensemble_size = validate_schedule_shapes(shape)

        t_base = torch.rand((batch_size, ensemble_size), device=device)
        t = t_base[:, None, :, None, None]

        return dict.fromkeys(shape, t)

    def build_training_pair(
        self,
        y: dict[str, torch.Tensor],
        schedule: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # Conditioning is an interpolated state y_t; target is velocity (eps - y).
        y_cond: dict[str, torch.Tensor] = {}
        target: dict[str, torch.Tensor] = {}
        for dataset_name, y_data in y.items():
            t = schedule[dataset_name]
            eps = torch.randn_like(y_data)
            y_t = (1.0 - t) * y_data + t * eps
            y_cond[dataset_name] = y_t
            target[dataset_name] = eps - y_data
        return y_cond, target

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
        # Model predicts velocity field given conditioning and t.
        return model(
            x,
            y_cond,
            schedule,
            model_comm_group=model_comm_group,
            grid_shard_shapes=grid_shard_shapes,
        )

    def clean_state(
        self,
        y_pred: dict[str, torch.Tensor],
        y_cond: dict[str, torch.Tensor],
        schedule: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        assert (
            y_cond is not None and schedule is not None
        ), "FlowObjective requires conditioning and schedule to compute clean state."

        # Reconstruct clean state from velocity prediction and interpolated input.
        recon: dict[str, torch.Tensor] = {}
        for dataset_name, y_t in y_cond.items():
            t = schedule[dataset_name]
            recon[dataset_name] = y_t - t * y_pred[dataset_name]
        return recon
