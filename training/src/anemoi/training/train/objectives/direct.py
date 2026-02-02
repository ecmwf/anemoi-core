# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from .base import ObjectiveStrategy


class DirectPredictionObjective(ObjectiveStrategy):
    """Direct prediction objective (no diffusion/flow noise)."""

    def sample_schedule(
        self,
        shape: dict[str, tuple[int, ...]],
        device: torch.device,
        *,
        model: torch.nn.Module | None = None,
    ) -> None:
        # No schedule needed for direct prediction.
        del shape, device, model
        return

    def build_training_pair(
        self,
        y: dict[str, torch.Tensor],
        schedule: None,
    ) -> tuple[None, dict[str, torch.Tensor]]:
        # Conditioning is unused; target is the clean output.
        del schedule
        return None, y

    def forward(
        self,
        model: torch.nn.Module,
        x: dict[str, torch.Tensor],
        y_cond: None,
        schedule: None,
        *,
        model_comm_group: object | None = None,
        grid_shard_shapes: dict[str, tuple[int, ...] | None] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        # Directly predict the clean output state.
        del y_cond, schedule
        return model(
            x,
            model_comm_group=model_comm_group,
            grid_shard_shapes=grid_shard_shapes,
            **kwargs,
        )
