# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.training.losses.scalers.base_scaler import BaseScaler
from anemoi.training.utils.enums import OutputTensorDim
import logging

import torch

LOGGER = logging.getLogger(__name__)


class LeadTimeDecayScaler(BaseScaler):
    """Class to weight each specific predicted lead time in the batch."""

    scale_dims = OutputTensorDim.TIME

    def __init__(
        self,
        relative_date_indices: list[int],
        decay_factor: float,
        method: str = "linear",
        inverse: bool = False,
        norm: str | None = None,
        **kwargs,
    ) -> None:
        """Initialise Scaler.

        Parameters
        ----------
        relative_date_indices : list[int]
           List of relative date indices for the target nodes. These are the indices of the lead times
           relative to the current time step, e.g. [0, 1, 2, ...] for a sequence of lead times.
        decay factor : float
           decay factor for the lead time weights computation. Higher values lead to faster decay.
        method : str, optional
            Method to use for the decay weights. Options are "exponential" and "linear", by default "linear".
        inverse : bool, optional
            Whether to use the inverse of the weights, by default False.
        norm : str, optional
            Type of normalization to apply. Options are None, unit-sum, unit-mean and l1.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(norm=norm)
        del kwargs
        assert method in ["exponential", "linear"], f"Method {method} not supported"
        self.method = method
        self.decay_factor = decay_factor
        self.inverse = inverse
        self.relative_date_indices = relative_date_indices

    def scale_forward(self) -> torch.Tensor:
        """Attributes the lead time weights to the target nodes."""
        if self.method == "exponential":
            return torch.exp(-self.decay_factor * torch.tensor(self.relative_date_indices))
        if self.method == "linear":
            return (
                1
                - self.decay_factor
                * torch.tensor(self.relative_date_indices)
                / torch.tensor(self.relative_date_indices).max()
            )
        msg = f"Method {self.method} not supported"
        raise NotImplementedError(msg)

    def scale_backward(self) -> torch.Tensor:
        """Same that forward weights but attributes an increasing weight to predicted tensors close to final lead time (e.g. max rollout or interpolator upper bound value)."""
        if self.method == "exponential":
            return 1 - torch.exp(-self.decay_factor * torch.tensor(self.relative_date_indices))
        if self.method == "linear":
            return (
                self.decay_factor
                * torch.tensor(self.relative_date_indices)
                / torch.tensor(self.relative_date_indices).max()
            )
        msg = f"Method {self.method} not supported"
        raise NotImplementedError(msg)

    def get_scaling_values(self) -> torch.Tensor:
        if self.inverse:
            return self.scale_backward()
        return self.scale_forward()
