# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import functools
import logging
from abc import ABC
from abc import abstractmethod

import torch
from torch import nn

from anemoi.training.losses.utils import ScaleTensor

LOGGER = logging.getLogger(__name__)


class BaseWeightedLoss(nn.Module, ABC):
    """Node-weighted general loss."""

    scalar: ScaleTensor

    def __init__(
        self,
        node_weights: torch.Tensor,
        ignore_nans: bool = False,
    ) -> None:
        """Node- and feature_weighted Loss.

        Exposes:
        - self.avg_function: torch.nanmean or torch.mean
        - self.sum_function: torch.nansum or torch.sum
        depending on the value of `ignore_nans`

        Registers:
        - self.node_weights: torch.Tensor of shape (N, )

        Parameters
        ----------
        node_weights : torch.Tensor of shape (N, )
            Weight of each node in the loss function
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False

        """
        super().__init__()

        self.scalar = ScaleTensor()

        self.avg_function = torch.nanmean if ignore_nans else torch.mean
        self.sum_function = torch.nansum if ignore_nans else torch.sum

        self.register_buffer("node_weights", node_weights, persistent=True)

    @functools.wraps(ScaleTensor.add_scalar, assigned=("__doc__", "__annotations__"))
    def add_scalar(self, dimension: int | tuple[int], scalar: torch.Tensor, *, name: str | None = None) -> None:
        self.scalar.add_scalar(dimension=dimension, scalar=scalar, name=name)

    def scale(
        self,
        x: torch.Tensor,
        feature_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scale a tensor by the variable_scaling.

        Parameters
        ----------
        x : torch.Tensor
            Tensor to be scaled, shape (bs, ensemble, lat*lon, n_outputs)
        feature_indices:
            feature indices (relative to full model output) of the features passed in pred and target

        Returns
        -------
        torch.Tensor
            Scaled error tensor
        """
        # Use feature_weights if available
        if len(self.scalar) == 0:
            return x

        scalar = self.scalar.get_scalar(x.ndim).to(x)

        if feature_indices is None:
            return x * scalar
        return x * scalar[..., feature_indices]

    def scale_by_node_weights(self, x: torch.Tensor, squash: bool = True) -> torch.Tensor:
        """Scale a tensor by the node_weights.

        Equivalent to reducing and averaging accordingly across all
        dimensions of the tensor.

        Parameters
        ----------
        x : torch.Tensor
            Tensor to be scaled, shape (bs, ensemble, lat*lon, n_outputs)
        squash : bool, optional
            Average last dimension, by default True
            If False, the loss returned of shape (n_outputs)

        Returns
        -------
        torch.Tensor
            Scaled error tensor
        """
        # Squash by last dimension
        if squash:
            x = self.avg_function(x, dim=-1)
            # Weight by area
            x *= self.node_weights.expand_as(x)
            x /= self.sum_function(self.node_weights.expand_as(x))
            return self.sum_function(x)

        # Weight by area, due to weighting construction is analagous to a mean
        x *= self.node_weights[..., None].expand_as(x)
        # keep last dimension (variables) when summing weights
        x /= self.sum_function(self.node_weights[..., None].expand_as(x), dim=(0, 1, 2))
        return self.sum_function(x, dim=(0, 1, 2))

    @abstractmethod
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        feature_indices: torch.Tensor | None = None,
        feature_scale: bool = True,
    ) -> torch.Tensor:
        """Calculates the lat-weighted scaled loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)
        squash : bool, optional
            Average last dimension, by default True
        feature_indices:
            feature indices (relative to full model output) of the features passed in pred and target
        feature_scale:
            If True, scale the loss by the feature_weights

        Returns
        -------
        torch.Tensor
            Weighted loss
        """
        out = pred - target

        if feature_scale:
            out = self.scale(out, feature_indices)
        return self.scale_by_node_weights(out, squash)

    @property
    def name(self) -> str:
        """Used for logging identification purposes."""
        return self.__class__.__name__.lower()


class FunctionalWeightedLoss(BaseWeightedLoss):
    """WeightedLoss which a user can subclass and provide `calculate_difference`."""

    def __init__(
        self,
        node_weights: torch.Tensor,
        ignore_nans: bool = False,
    ) -> None:
        super().__init__(node_weights, ignore_nans)

    @abstractmethod
    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate Difference between prediction and target."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        feature_indices: torch.Tensor | None = None,
        feature_scale: bool = True,
    ) -> torch.Tensor:
        """Calculates the lat-weighted scaled loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)
        squash : bool, optional
            Average last dimension, by default True
        feature_indices:
            feature indices (relative to full model output) of the features passed in pred and target
        feature_scale:
            If True, scale the loss by the feature_weights

        Returns
        -------
        torch.Tensor
            Weighted loss
        """
        out = self.calculate_difference(pred, target)

        if feature_scale:
            out = self.scale(out, feature_indices)
        return self.scale_by_node_weights(out, squash)
