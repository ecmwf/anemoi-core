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
from typing import TYPE_CHECKING

from anemoi.training.losses.mse import MSELoss

if TYPE_CHECKING:
    import torch

LOGGER = logging.getLogger(__name__)


class LimitedAreaMSELoss(MSELoss):
    """MSE loss, calculated only within or outside the limited area.

    Further, the loss can be computed for the specified region (default),
    or as the contribution to the overall loss.
    """

    name = "mse"

    def __init__(
        self,
        lam_scalar: str,
        wmse_contribution: bool = False,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        """Node- and feature weighted MSE Loss.

        Parameters
        ----------
        wmse_contribution: bool
            compute loss as the contribution to the overall MSE, by default False
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False
        """
        self.lam_scalar = lam_scalar
        super().__init__(ignore_nans=ignore_nans, **kwargs)
        self.wmse_contribution = wmse_contribution

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        scaler_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calculates the lat-weighted MSE loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)
        squash : bool, optional
            Average last dimension, by default True
        scaler_indices:
            feature indices (relative to full model output) of the features passed in pred and target

        Returns
        -------
        torch.Tensor
            Weighted MSE loss
        """
        out = self.calculate_difference(pred, target)

        limited_area_mask = self.scaler.subset(self.lam_scalar).get_scaler(out.ndim, out.device).bool()

        out *= limited_area_mask

        out = self.scale(out, scaler_indices, without_scalers=[self.lam_scalar])

        # TODO(Mario): Fix contribution

        if squash:
            out = self.avg_function(out, dim=-1)

        return self.sum_function(out, dim=(0, 1, 2))
