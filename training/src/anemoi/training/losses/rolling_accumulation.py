# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
import torch.nn.functional as F

from anemoi.training.losses.base import FunctionalLoss


class RollingAccumulationHuberLoss(FunctionalLoss):
    """Huber loss on short-window rolling accumulations."""

    name: str = "rolling_accumulation_huber"

    def __init__(
        self,
        window_size: int,
        step_seconds: float,
        delta: float = 1.0,
        rate_to_amount: bool = True,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans, **kwargs)
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if step_seconds <= 0:
            raise ValueError("step_seconds must be > 0")
        self.window_size = int(window_size)
        self.step_seconds = float(step_seconds)
        self.delta = float(delta)
        self.rate_to_amount = bool(rate_to_amount)

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim != 5 or target.ndim != 5:
            raise ValueError("Expected pred and target to have shape (batch, time, ensemble, grid, vars)")
        if pred.shape[1] < self.window_size or target.shape[1] < self.window_size:
            raise ValueError(
                f"window_size={self.window_size} exceeds available output steps "
                f"(pred={pred.shape[1]}, target={target.shape[1]})"
            )

        if self.rate_to_amount:
            time_factor = self.step_seconds / 3600.0
            pred = pred * time_factor
            target = target * time_factor

        pred_acc = pred.unfold(dimension=1, size=self.window_size, step=1).sum(dim=-1)
        target_acc = target.unfold(dimension=1, size=self.window_size, step=1).sum(dim=-1)

        return F.huber_loss(pred_acc, target_acc, reduction="none", delta=self.delta)
