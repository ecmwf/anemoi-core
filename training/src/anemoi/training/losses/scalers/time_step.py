# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import torch

from anemoi.training.losses.scalers.base_scaler import BaseScaler
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


class TimeStepScaler(BaseScaler):
    """Class for scaling different output step contributions to the loss."""

    scale_dims: TensorDim = TensorDim.TIME

    def __init__(
        self,
        weights: torch.Tensor,
        norm: str | None = None,
        **kwargs,
    ) -> None:
        """Initialise Scaler.

        Parameters
        ----------
        weights: list
            Weight per output step.
        norm : str, optional
            Type of normalization to apply. Options are None, unit-sum, unit-mean and l1.
        """
        super().__init__(norm=norm)
        del kwargs
        self.weights = weights

    def get_scaling_values(self) -> torch.Tensor:
        return torch.tensor(self.weights, dtype=torch.float32)
