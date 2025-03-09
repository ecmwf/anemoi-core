# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations
from functools import cached_property
import logging
from typing import Union, Callable

import torch
from torch import nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)

class HingeLoss(nn.Module):
    """Latitude- and feature-weighted hinge loss.
    
    This loss computes the hinge loss between predictions and targets,
    applying node and feature weights. The hinge loss is defined as
    max(0, margin - y * f(x)) where y is the target and f(x) is the prediction.
    """

    def __init__(
        self,
        node_weights: torch.Tensor,
        feature_weights: torch.Tensor,
        margin: float = 1.0,
        ignore_nans: bool | None = False,
        patch_proportion: float = 0.15,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        node_weights : torch.Tensor of shape (N, )
            Weight of each node in the loss function.
        feature_weights : torch.Tensor of shape (F, )
            Weight of each feature in the loss function.
        margin : float, optional
            The margin in the hinge loss formula, by default 1.0
        ignore_nans : bool, optional
            If True, the loss functions will use nanmean/nansum to ignore NaNs,
            by default False.
        """
        super().__init__()

        self.avg_function = torch.nanmean if ignore_nans else torch.mean
        self.sum_function = torch.nansum if ignore_nans else torch.sum
        self.margin = margin

        self.patch_proportion = patch_proportion

        # We add an extra dimension to node_weights for broadcasting
        self.register_buffer("node_weights", node_weights[..., None], persistent=True)
        self.register_buffer("feature_weights", feature_weights, persistent=True)



    def forward(
        self,
        fake_outp: torch.Tensor,
        real_outp: torch.Tensor | None = None,
        squash: Union[bool, tuple] = True,
        feature_scale: bool = True,
        feature_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes the lat-weighted hinge loss.
        
        Parameters
        ----------
        fake_outp : torch.Tensor
            Fake/generated output tensor, shape (bs, timesteps, ens, lat*lon, n_features)
        real_outp : torch.Tensor | None
            Real output tensor, shape (bs, timesteps, ens, lat*lon, n_features)
            Should contain values of -1 or 1. If None, only fake loss is computed.
        squash : bool or tuple, optional
            Whether to reduce (sum) over the spatial and feature dimensions,
            by default True.
        feature_scale : bool, optional
            Whether to scale the loss by the feature weights,
            by default True.
        feature_indices : torch.Tensor or None, optional
            Indices of the features to scale the loss by, by default None.
        
        Returns
        -------
        torch.Tensor
            The computed hinge loss.
        """
        # Compute hinge loss for real and fake outputs
        if real_outp is not None:
            losses_real = F.relu(self.margin - real_outp)  # real samples should be classified as positive (close to 1)
            losses_fake = F.relu(self.margin + fake_outp)  # fake samples should be classified as negative (close to -1)
            
            # Combine the losses
            losses = (losses_real + losses_fake) / 2
        else:
            losses = F.relu(self.margin + fake_outp)
        
        # Apply feature weighting if enabled
        if feature_scale:
            if feature_indices is not None:
                # If we're only interested in specific features, select them for weighting
                feature_weights = self.feature_weights[feature_indices]
                losses = losses * feature_weights
                losses = losses / feature_weights.sum()
            else:
                losses = losses * self.feature_weights
                losses = losses / self.feature_weights.sum()
        
        # Apply node weighting over spatial dimensions
        losses = losses * (self.node_weights / self.sum_function(self.node_weights))
        
        # Normalize by ensemble dimension (assumed to be axis -3)
        losses = losses / losses.shape[-3]
        
        # Normalize by batch dimension (assumed to be axis -5)
        losses = losses / losses.shape[-5]

        # Taking a random patch of the loss (patching applied in the spatial & feature dimensions)
        # Set (1 - patch proportion) of the values to 0 
        # Then re-weight the loss values by the inverse of the patch proportion
        if self.patch_proportion > 0:
            rand_mask = torch.rand_like(losses)
            rand_mask = rand_mask < self.patch_proportion
            losses = losses * rand_mask
            losses = losses / torch.maximum(rand_mask.sum() / rand_mask.numel(), torch.tensor(1e-6, device=losses.device))
        
        # Optionally squash over spatial and feature dimensions
        if squash:
            # If squash is a tuple, use it directly; otherwise, assume these axes: batch, ensemble, time, spatial, feature
            axes = squash if isinstance(squash, tuple) else (-5, -3, -2, -1)
            losses = self.sum_function(losses, axis=axes)
        
        return losses

    @cached_property
    def name(self) -> str:
        """Returns the name of the loss for logging."""
        return "hinge"