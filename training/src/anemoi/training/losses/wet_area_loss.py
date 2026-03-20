# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch

from anemoi.training.losses.base import FunctionalLoss


class WeightedSoftWetAreaLoss(FunctionalLoss):
    name: str = "weighted_soft_wet_area"

    def __init__(
        self,
        threshold: float = 0.0,
        temperature: float = 0.1,
        false_positive_weight: float = 1.5,
        false_negative_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.false_positive_weight = float(false_positive_weight)
        self.false_negative_weight = float(false_negative_weight)

    def _soft_wet_mask(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((x - self.threshold) / self.temperature)

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_wet = self._soft_wet_mask(pred)
        target_wet = self._soft_wet_mask(target)

        diff = pred_wet - target_wet
        fp = torch.relu(diff)
        fn = torch.relu(-diff)

        return self.false_positive_weight * fp**2 + self.false_negative_weight * fn**2
