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

import numpy as np
import torch

from anemoi.training.losses.weightedloss import BaseWeightedLoss

LOGGER = logging.getLogger(__name__)


class LogCosh(torch.autograd.Function):
    """LogCosh custom autograd function."""

    @staticmethod
    def forward(ctx, inp: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
        ctx.save_for_backward(inp)
        abs_input = torch.abs(inp)
        return abs_input + torch.nn.functional.softplus(-2 * abs_input) - np.log(2)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
        (inp,) = ctx.saved_tensors
        return grad_output * torch.tanh(inp)


class WeightedLogCoshLoss(BaseWeightedLoss):
    """Node-weighted LogCosh loss."""

    name = "wlogcosh"

    def __init__(
        self,
        node_weights: torch.Tensor,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        """Node- and feature weighted LogCosh Loss.

        Parameters
        ----------
        node_weights : torch.Tensor of shape (N, )
            Weight of each node in the loss function
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False

        """
        super().__init__(
            node_weights=node_weights,
            ignore_nans=ignore_nans,
            **kwargs,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        scalar_indices: tuple[int, ...] | None = None,
        without_scalars: list[str] | list[int] | None = None,
    ) -> torch.Tensor:
        """Calculates the lat-weighted LogCosh loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)
        squash : bool, optional
            Average last dimension, by default True
        scalar_indices: tuple[int,...], optional
            Indices to subset the calculated scalar with, by default None
        without_scalars: list[str] | list[int] | None, optional
            list of scalars to exclude from scaling. Can be list of names or dimensions to exclude.
            By default None

        Returns
        -------
        torch.Tensor
            Weighted LogCosh loss

        """
        out = LogCosh.apply(pred - target)
        out = self.scale(out, scalar_indices, without_scalars=without_scalars)
        return self.scale_by_node_weights(out, squash)
