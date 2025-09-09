# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from collections.abc import Callable

from anemoi.training.losses.base import FunctionalLoss

LOGGER = logging.getLogger(__name__)


class TorchLoss(FunctionalLoss):
    """Loss function."""

    def __init__(self, loss: Callable, ignore_nans: bool = False, **_kwargs) -> None:
        super().__init__(ignore_nans)
        self.loss = loss
        self.name = loss.__class__.__name__.lower().replace("loss", "")

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, gridpoints, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, gridpoints, n_outputs)

        Returns
        -------
        torch.Tensor
            Loss
        """
        return self.loss(pred, target)
