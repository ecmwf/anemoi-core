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

import torch

from anemoi.training.losses.weightedloss import BaseWeightedLoss

LOGGER = logging.getLogger(__name__)


class WeightedMSELossLimitedArea(BaseWeightedLoss):
    """Node-weighted MSE loss, calculated only within or outside the limited area.

    Further, the loss can be computed for the specified region (default),
    or as the contribution to the overall loss.
    """

    name = "wmse"

    def __init__(
        self,
        node_weights: torch.Tensor,
        inside_lam: bool = True,
        wmse_contribution: bool = False,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        """Node- and feature weighted MSE Loss.

        Parameters
        ----------
        node_weights : torch.Tensor of shape (N, )
            Weight of each node in the loss function
        mask: torch.Tensor
            the mask marking the indices of the regional data points (bool)
        inside_lam: bool
            compute the loss inside or outside the limited area, by default inside (True)
        wmse_contribution: bool
            compute loss as the contribution to the overall MSE, by default False
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False
        """
        super().__init__(
            node_weights=node_weights,
            ignore_nans=ignore_nans,
            **kwargs,
        )

        self.inside_lam = inside_lam
        self.wmse_contribution = wmse_contribution

        if inside_lam:
            self.name += "_inside_lam"
        else:
            self.name += "_outside_lam"
        if wmse_contribution:
            self.name += "_contribution"

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        scalar_indices: torch.Tensor | None = None,
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
        scalar_indices:
            feature indices (relative to full model output) of the features passed in pred and target

        Returns
        -------
        torch.Tensor
            Weighted MSE loss
        """
        out = torch.square(pred - target)

        limited_area_mask = self.scalar.subset("limited_area_mask").get_scalar(out.ndim, out.device)

        if not self.inside_lam:
            limited_area_mask = ~limited_area_mask

        if not self.wmse_contribution:
            self.node_weights *= limited_area_mask[0, 0, :, 0]

        out *= limited_area_mask

        out = self.scale(out, scalar_indices, without_scalars=["limited_area_mask"])

        return self.scale_by_node_weights(out, squash)
