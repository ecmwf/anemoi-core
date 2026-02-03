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

from anemoi.training.losses.base import FunctionalLoss

LOGGER = logging.getLogger(__name__)


class HuberLoss(FunctionalLoss):
    """Huber loss."""

    name: str = "huber"

    def __init__(
        self,
        delta: float = 1.0,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        """Node- and feature weighted Huber Loss.

        See `Huber loss <https://en.wikipedia.org/wiki/Huber_loss>`_ for more information.

        Parameters
        ----------
        delta : float, optional
            Threshold for Huber loss, by default 1.0
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False
        """
        super().__init__(ignore_nans=ignore_nans, **kwargs)
        self.delta = delta

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the Huber loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)

        Returns
        -------
        torch.Tensor
            Huber loss
        """
        diff = torch.abs(pred - target)
        return torch.where(diff < self.delta, 0.5 * torch.square(diff), self.delta * (diff - 0.5 * self.delta))
