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

import torch
from torch import nn
LOGGER = logging.getLogger(__name__)
from typing import Union
from torch.nn import Module
from omegaconf import DictConfig, ListConfig
from anemoi.training.utils.debug_hydra import instantiate_debug
# TODO(rilwan-ade): make parent loss calss that holds the common methods avg_function and sum_function
import einops

class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for GAN training.
    
    Computes the distance between discriminator features from real and fake data 
    across multiple layers, with configurable weighting.
    """

    def __init__(
        self,
        loss_func: Module | DictConfig,
        node_weights: torch.Tensor,
        feature_weights: torch.Tensor,
        discriminator_layers_count: int,
        layer_weights: list[float],
        ignore_nans: bool | None = False,
        patch_proportion: float = 0.00,
        **kwargs,
    ) -> None:
        """Feature matching loss using distance between discriminator features.

        Parameters
        ----------
        loss_func : Module | DictConfig
            Base loss function to compute distance between features
        node_weights : torch.Tensor of shape (N, )
            Weight of each node in the loss function
        feature_weights : torch.Tensor of shape (F, )
            Weight of each feature in the loss function
        discriminator_layers_count : int
            Number of layers in the discriminator that will provide features
        layer_weights : list[float]
            Weights for each discriminator layer
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans, by default False
        patch_proportion : float, optional
            Proportion of the loss to randomly sample, by default 0.00
        """
        super().__init__()

        self.avg_function = torch.nanmean if ignore_nans else torch.mean
        self.sum_function = torch.nansum if ignore_nans else torch.sum

        if isinstance(loss_func, DictConfig):
            self.loss_func = instantiate_debug(loss_func, node_weights=node_weights, feature_weights=feature_weights)
        else:
            self.loss_func = loss_func

        # Process layer weights
        if isinstance(layer_weights, (list, ListConfig)):
            assert len(layer_weights) == discriminator_layers_count, (
                f"Number of layer weights ({len(layer_weights)}) must match the number of "
                f"discriminator layers ({discriminator_layers_count})"
            )
            weights_tensor = torch.as_tensor(layer_weights)
            normalized_layer_weights = weights_tensor / torch.sum(weights_tensor)
            normalized_layer_weights = normalized_layer_weights[:, None, None, None, None, None]
        else:
            normalized_layer_weights = self.get_layer_weights(layer_weights, discriminator_layers_count)

        self.register_buffer("layer_weights", normalized_layer_weights, persistent=True)
        self.patch_proportion = patch_proportion

    def get_layer_weights(self, weighting_scheme: str, discriminator_layers_count: int) -> torch.Tensor:
        """Generate layer weights based on a named weighting scheme.

        Parameters
        ----------
        weighting_scheme : str
            Name of the weighting scheme to use
        discriminator_layers_count : int
            Number of discriminator layers

        Returns
        -------
        torch.Tensor
            Tensor of layer weights

        Raises
        ------
        NotImplementedError
            If the weighting scheme is not implemented
        """
        raise NotImplementedError(f"Layer weighting scheme '{weighting_scheme}' not implemented")

    def forward(
        self,
        features_real: list[torch.Tensor],
        features_fake: list[torch.Tensor],
        squash: Union[bool, tuple] = True,
        feature_scale: bool = True,
        feature_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Calculates the feature matching loss between real and fake feature maps.

        Parameters
        ----------
        features_real : list[torch.Tensor] 
            List of feature tensors from discriminator for real data
        features_fake : list[torch.Tensor]
            List of feature tensors from discriminator for fake data
        squash : bool | tuple, optional
            Whether to reduce (sum) over dimensions, by default True
        feature_scale : bool, optional
            Whether to scale the loss by the feature weights, by default True
        feature_indices : torch.Tensor | None, optional
            Indices of the features to scale the loss by, by default None

        Returns
        -------
        torch.Tensor
            Feature matching loss
        """
        # Initialize tensor to store losses for each layer with the same shape as features
        layer_losses = torch.zeros(
            (len(features_real), *features_real[0].shape), 
            device=features_real[0].device, 
            dtype=features_real[0].dtype
        )

        # Calculate loss for each layer
        for i, (real_feat, fake_feat) in enumerate(zip(features_real, features_fake)):
            # Calculate loss without reducing dimensions or applying feature scaling yet
            layer_loss = self.loss_func(fake_feat, real_feat, squash=False, feature_scale=False)
            layer_losses[i] = layer_loss

        # Apply feature weighting if enabled
        if feature_scale:
            if feature_indices is not None:
                # Apply weighting only to selected features
                feature_weights = self.loss_func.feature_weights[feature_indices]
                layer_losses = layer_losses * feature_weights
                layer_losses = layer_losses / feature_weights.sum()
            else:
                layer_losses = layer_losses * self.loss_func.feature_weights
                layer_losses = layer_losses / self.loss_func.feature_weights.sum()
        
        # Apply node weighting over spatial dimensions
        layer_losses = layer_losses * (self.loss_func.node_weights / self.loss_func.sum_function(self.loss_func.node_weights))
        
        # Normalize by ensemble dimension (assumed to be axis -3)
        layer_losses = layer_losses / layer_losses.shape[-3]
        
        # Normalize by batch dimension (assumed to be axis -5)
        layer_losses = layer_losses / layer_losses.shape[-5]

        # Apply layer weighting
        layer_losses = layer_losses * self.layer_weights

        # Taking a random patch of the loss if patch_proportion > 0
        if self.patch_proportion > 0:
            rand_mask = torch.rand_like(layer_losses)
            rand_mask = rand_mask < self.patch_proportion
            layer_losses = layer_losses * rand_mask
            layer_losses = layer_losses / torch.maximum(
                rand_mask.sum() / rand_mask.numel(), 
                torch.tensor(1e-6, device=layer_losses.device)
            )

        # Sum across layers
        layer_losses = self.sum_function(layer_losses, axis=0)

        # Optionally reduce across other dimensions
        if squash:
            # If squash is a tuple, use it directly; otherwise, use default dimensions
            axes = squash if isinstance(squash, tuple) else (-5, -3, -2, -1)
            layer_losses = self.sum_function(layer_losses, axis=axes)
        
        return layer_losses

    @cached_property
    def name(self) -> str:
        """Returns the name of the loss for logging."""
        return "feature_matching"
