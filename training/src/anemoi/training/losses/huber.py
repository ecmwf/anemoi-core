# (C) Copyright 2024-2026 Anemoi contributors.
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


class ReversedHuberLoss(FunctionalLoss):
    """Reversed Huber loss.

    Linear for small errors and quadratic for large errors, i.e. the opposite
    of the classic Huber loss. Robust to small-scale noise (MAE-like) while
    strongly penalising catastrophic forecast failures (MSE-like).

    Smooth form::

        w(e) = sigmoid(sharpness * (|e| - delta))
        L(e) = (1 - w) * delta * |e| + w * 0.5 * e**2

    Exact piecewise form (``smooth=False``)::

        L(e) = delta * |e|                   if |e| <= delta
               0.5 * e**2 + 0.5 * delta**2   otherwise

    The piecewise form is continuous at ``|e| = delta`` but its gradient is not.

    Parameters
    ----------
    delta : float, optional
        Transition point between linear and quadratic regimes, in normalised
        units. Default 1.0
    sharpness : float, optional
        Steepness of the sigmoid blend. Defaults to 2.0.
    smooth : bool, optional
        Use the sigmoid-blended form (True) or the exact piecewise form (False),
        by default True.
    ignore_nans : bool, optional
        Allow NaNs in the target, by default False.
    """

    def __init__(
        self,
        delta: float = 1.0,
        sharpness: float = 2.0,
        smooth: bool = True,
        ignore_nans: bool = False,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans)
        if delta <= 0:
            msg = f"delta must be positive, got {delta}"
            raise ValueError(msg)
        if sharpness <= 0:
            msg = f"sharpness must be positive, got {sharpness}"
            raise ValueError(msg)
        self.delta = float(delta)
        self.sharpness = float(sharpness)
        self.smooth = smooth

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Element-wise reversed Huber loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)

        Returns
        -------
        torch.Tensor
            Reversed Huber loss
        """
        err = (pred - target).float()
        abs_err = err.abs()
        linear = self.delta * abs_err
        quadratic = 0.5 * err.square()

        if self.smooth:
            w = torch.sigmoid(self.sharpness * (abs_err - self.delta))
            return torch.lerp(linear, quadratic, w)  # (1 - w) * linear + w * quadratic

        return torch.where(abs_err <= self.delta, linear, quadratic + 0.5 * self.delta**2)