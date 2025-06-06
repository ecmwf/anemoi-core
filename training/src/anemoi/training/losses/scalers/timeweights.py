# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

import logging
from typing import Optional

import torch

from anemoi.training.losses.scalers.base_scaler import TimeVaryingScaler
from anemoi.training.utils.enums import OutputTensorDim

LOGGER = logging.getLogger(__name__)


class LeadTimeScaler(TimeVaryingScaler):
    """Class to weight each specific predicted lead time in the batch."""

    scale_dims = OutputTensorDim.TIME

    def __init__(
        self,
        decay_factor: float,
        max_lead_time: int,
        decay_type: str = "linear",
        inverse: bool = False,
        norm: str | None = None,
        **kwargs,
    ) -> None:
        """Initialise Scaler.

        Parameters
        ----------
        decay_factor : float
           Decay factor for the lead time weights computation. Higher values lead to faster decay.
        max_lead_time : int
           Max predicted lead time relative to the current time step, i.e. 6 for 6 steps forwards.
        decay_type : str, optional
            Decay type to use for the decay weights. Options are "exponential" and "linear", by default "linear".
        inverse : bool, optional
            Whether to use the inverse of the weights, by default False.
        norm : str, optional
            Type of normalization to apply. Options are None, unit-sum, unit-mean and l1.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(norm=norm)
        del kwargs
        assert decay_type in ["exponential", "linear"], f"decay_type {decay_type} not supported"
        self.decay_type = decay_type
        self.decay_factor = decay_factor
        self.inverse = inverse
        self.max_lead_time = max_lead_time

    def scale_forward(self, lead_time: int) -> torch.Tensor:
        """Attributes the lead time weights to the target nodes."""
        if self.decay_type == "exponential":
            return torch.exp(-self.decay_factor * torch.tensor(lead_time))
        if self.decay_type == "linear":
            return 1 - self.decay_factor * torch.tensor(lead_time) / self.max_lead_time
        msg = f"decay_type {self.decay_type} not supported"
        raise NotImplementedError(msg)

    def scale_backward(self, lead_time: int) -> torch.Tensor:
        """Same as forward weights but attributes an increasing weight to predicted tensors close to final lead time."""
        if self.decay_type == "exponential":
            return 1 - torch.exp(-self.decay_factor * torch.tensor(lead_time))
        if self.decay_type == "linear":
            return self.decay_factor * torch.tensor(lead_time) / self.max_lead_time
        msg = f"decay_type {self.decay_type} not supported"
        raise NotImplementedError(msg)

    def get_scaling_values(self, lead_time: Optional[int] = None) -> torch.Tensor:
        if lead_time is None:
            return torch.tensor([1])
        if self.inverse:
            return self.scale_backward(lead_time)
        return torch.tensor([self.scale_forward(lead_time)])
