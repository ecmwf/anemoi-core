# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class ObjectiveStrategy(ABC):
    """Strategy interface for training objectives (deterministic/diffusion/flow)."""

    @abstractmethod
    def sample_schedule(
        self,
        shape: dict[str, tuple[int, ...]],
        device: torch.device,
        *,
        model: torch.nn.Module | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Sample a per-batch schedule (e.g. sigma or time)."""

    @abstractmethod
    def build_training_pair(
        self,
        y: dict[str, torch.Tensor],
        schedule: dict[str, torch.Tensor] | None,
    ) -> tuple[dict[str, torch.Tensor] | None, dict[str, torch.Tensor]]:
        """Build (conditioning input, target) for the objective.

        Notes
        -----
        - y is the clean target state/tendency in normalized space.
        - target is the objective-space loss target (may differ from y).
          Examples: direct target=y; diffusion target=y with noised y_cond;
          flow target=eps-y (velocity).
        """

    @abstractmethod
    def forward(
        self,
        model: torch.nn.Module,
        x: dict[str, torch.Tensor],
        y_cond: dict[str, torch.Tensor] | None,
        schedule: dict[str, torch.Tensor] | None,
        *,
        model_comm_group: object | None = None,
        grid_shard_shapes: dict[str, tuple[int, ...] | None] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for the objective."""

    def pre_loss_weights(
        self,
        schedule: dict[str, torch.Tensor] | None,
        *,
        model: torch.nn.Module | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Optional weights applied to the raw loss difference before scaling."""
        del schedule, model
        return None

    def clean_state(
        self,
        y_pred: dict[str, torch.Tensor],
        y_cond: dict[str, torch.Tensor] | None,
        schedule: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor] | None:
        """Return the state to use for rollout advancement."""
        del y_cond, schedule
        return y_pred

    def clean_pred_target_pair(
        self,
        y_pred: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
        y_cond: dict[str, torch.Tensor] | None,
        schedule: dict[str, torch.Tensor] | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Return (clean prediction, clean target) tensors for metrics and rollout.

        Note: callers must pass the clean target `y` (not the objective-space loss target).
        """
        y_recon = self.clean_state(y_pred, y_cond, schedule)
        assert y_recon is not None, "Objective must return a clean state for metrics."
        return y_recon, y


def validate_schedule_shapes(shape: dict[str, tuple[int, ...]]) -> tuple[int, int]:
    dataset_names = list(shape.keys())
    assert dataset_names, "Expected at least one dataset to infer schedule shapes."

    ref_shape = shape[dataset_names[0]]
    msg = "Expected 5D tensor shape (batch, time, ensemble, grid, vars) for objective schedules."
    assert len(ref_shape) == 5, msg

    batch_size = ref_shape[0]
    ensemble_size = ref_shape[2]

    for dataset_name, shape_x in shape.items():
        if len(shape_x) != 5:
            msg = f"Expected 5D tensor shape (batch, time, ensemble, grid, vars) for dataset '{dataset_name}'."
            raise AssertionError(msg)
        if shape_x[0] != batch_size or shape_x[2] != ensemble_size:
            msg = "Batch or ensemble dimension mismatch across datasets when sampling schedules."
            raise AssertionError(msg)

    return batch_size, ensemble_size
