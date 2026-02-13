# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import numpy as np
import torch

from anemoi.training.losses.base import FunctionalLoss

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


class LogCoshLoss(FunctionalLoss):
    """LogCosh loss."""

    name: str = "logcosh"

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the Log-cosh loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)

        Returns
        -------
        torch.Tensor
            Log-cosh loss
        """
        return LogCosh.apply(pred - target)
