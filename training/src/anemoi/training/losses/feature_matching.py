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
from anemoi.training.losses.weightedloss import BaseWeightedLoss
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

        # Initialize the underlying loss function
        if isinstance(loss_func, DictConfig):
            self.loss_func = instantiate_debug(
                loss_func, 
                node_weights=node_weights, 
                feature_weights=feature_weights,
                ignore_nans=ignore_nans,
                **kwargs
            )
        else:
            self.loss_func = loss_func
            
        # Verify the loss function has the expected interface
        if not isinstance(self.loss_func, BaseWeightedLoss):
            raise TypeError(f"loss_func must be a BaseWeightedLoss, got {type(self.loss_func)}")

        # Set up helper functions from the contained loss
        self.avg_function = self.loss_func.avg_function
        self.sum_function = self.loss_func.sum_function

        # Process layer weights
        if isinstance(layer_weights, (list, ListConfig)):
            assert len(layer_weights) == discriminator_layers_count, (
                f"Number of layer weights ({len(layer_weights)}) must match the number of "
                f"discriminator layers ({discriminator_layers_count})"
            )
            weights_tensor = torch.as_tensor(layer_weights)
            normalized_layer_weights = weights_tensor / torch.sum(weights_tensor)
            # Reshape for broadcasting: [layers, 1, 1, 1, 1, 1]
            normalized_layer_weights = normalized_layer_weights.view(-1, 1, 1, 1, 1, 1)
        else:
            normalized_layer_weights = torch.ones(discriminator_layers_count) / discriminator_layers_count
            normalized_layer_weights = normalized_layer_weights.view(-1, 1, 1, 1, 1, 1)

        self.register_buffer("layer_weights", normalized_layer_weights, persistent=True)
        self.patch_proportion = patch_proportion

    def forward(
        self,
        features_real: list[torch.Tensor],
        features_fake: list[torch.Tensor],
        squash: Union[bool, tuple] = True,
        feature_scale: bool = True,
        feature_indices: torch.Tensor | None = None,
        scalar_indices: tuple[int, ...] | None = None,
        without_scalars: list[str] | list[int] | None = None,
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
            Whether to scale the loss by feature weights, by default True
        feature_indices : torch.Tensor | None, optional
            Indices of the features to scale the loss by, by default None
        scalar_indices : tuple[int, ...] | None, optional
            Indices to use for scaling, by default None
        without_scalars : list[str] | list[int] | None, optional
            List of scalars to exclude from scaling, by default None

        Returns
        -------
        torch.Tensor
            Feature matching loss
        """
        # Verify inputs
        assert len(features_real) == len(features_fake), "Real and fake feature lists must have same length"
        assert len(features_real) > 0, "Feature lists cannot be empty"
        
        # Make sure features are on the same device as layer weights
        device = self.layer_weights.device
        
        # Initialize tensor to store losses for each layer with the same shape as features
        layer_losses = []
        
        # Calculate loss for each layer using the underlying loss function
        for i, (real_feat, fake_feat) in enumerate(zip(features_real, features_fake)):
            # Ensure features are on the correct device
            real_feat = real_feat.to(device)
            fake_feat = fake_feat.to(device)
            
            # Use the contained loss function, deferring squashing/scaling to later
            layer_loss = self.loss_func(
                fake_feat, 
                real_feat, 
                squash=False,  # Don't squash yet, we need to apply layer weights first
                feature_scale=feature_scale,
                feature_indices=feature_indices,
                scalar_indices=scalar_indices,
                without_scalars=without_scalars
            )
            
            # Apply layer weighting - ensure correct broadcasting
            weighted_loss = layer_loss * self.layer_weights[i]
            layer_losses.append(weighted_loss)
            
        # Stack losses from different layers
        if all(loss.dim() == layer_losses[0].dim() for loss in layer_losses):
            layer_losses = torch.stack(layer_losses, dim=0)
        else:
            # Handle case where losses have different dimensions
            LOGGER.warning("Layer losses have different dimensions, summing individually")
            total_loss = sum(loss.sum() for loss in layer_losses)
            return total_loss

        # Taking a random patch of the loss if patch_proportion > 0
        if self.patch_proportion > 0:
            rand_mask = torch.rand_like(layer_losses)
            rand_mask = rand_mask < self.patch_proportion
            layer_losses = layer_losses * rand_mask
            layer_losses = layer_losses / torch.maximum(
                rand_mask.sum() / rand_mask.numel(), 
                torch.tensor(1e-6, device=layer_losses.device)
            )

        
        layer_losses = self.sum_function(layer_losses)
        
        return layer_losses

    @cached_property
    def name(self) -> str:
        """Returns the name of the loss for logging."""
        return f"feature_matching_{self.loss_func.name}"
