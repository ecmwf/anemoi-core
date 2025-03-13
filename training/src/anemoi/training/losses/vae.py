from __future__ import annotations

import logging
from functools import cached_property

import torch
from torch import nn
from torch import Tensor
from omegaconf import DictConfig
from typing import Optional, Union, Dict

from anemoi.training.losses.weightedloss import BaseWeightedLoss
from anemoi.training.utils.debug_hydra import instantiate_debug

LOGGER = logging.getLogger(__name__)

class VAELoss(BaseWeightedLoss):
    """Variational Autoencoder loss, combining reconstruction loss and KL divergence loss.
    
    This class computes the combined VAE loss with configurable weighting between
    the reconstruction term and the KL divergence term.
    """

    def __init__(
        self,
        node_weights: Tensor,
        feature_weights: Tensor,
        reconstruction_loss: dict | DictConfig | nn.Module,
        divergence_loss: dict | DictConfig | nn.Module,
        latent_node_weights: Tensor,
        divergence_loss_weight: Tensor | float = 1e-2,
        # discriminator_loss: dict | DictConfig | nn.Module | None = None,
        # adversarial_loss_weight: Tensor | float = 1e-2,
        # feature_matching_loss: dict | DictConfig | nn.Module | None = None, #TODO: this will be the self contained version of the perceptual loss
        # feature_matching_loss_weight: Tensor | float = 1e-2,
        ignore_nans: bool | None = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        node_weights : Tensor
            Weights for each spatial node in the grid, shape (lat*lon, 1)
        feature_weights : Tensor
            Weights for each feature, shape (n_features,)
        reconstruction_loss : dict | DictConfig | nn.Module
            Loss function for reconstruction term
        divergence_loss : dict | DictConfig | nn.Module
            Loss function for divergence (KL) term
        latent_node_weights : Tensor
            Weights for each spatial node in the latent space
        divergence_loss_weight : Tensor | float, optional
            Weight factor for the divergence loss, by default 1e-2
        ignore_nans : bool | None, optional
            If True, use nanmean/nansum to ignore NaNs in loss computation, by default False
        """
        super().__init__(
            node_weights=node_weights,
            ignore_nans=ignore_nans,
            **kwargs,
        )

        self.avg_function = torch.nanmean if ignore_nans else torch.mean
        self.sum_function = torch.nansum if ignore_nans else torch.sum

        # Initialize reconstruction loss
        if isinstance(reconstruction_loss, (dict, DictConfig)):
            self.reconstruction_loss = instantiate_debug(
                reconstruction_loss,
                node_weights=node_weights,
                feature_weights=feature_weights,
                ignore_nans=ignore_nans,
                **kwargs,
            )
        elif isinstance(reconstruction_loss, nn.Module):
            self.reconstruction_loss = reconstruction_loss
        else:
            raise ValueError(f"Invalid reconstruction loss: {reconstruction_loss}")

        # Initialize divergence loss
        if isinstance(divergence_loss, (dict, DictConfig)):
            # A similar scaling must be used for the divergence loss as is used for the reconstruction
            # the latent area weights are already handled (when they are normalized)
            self.divergence_loss = instantiate_debug(
                divergence_loss,
                node_weights=latent_node_weights,
                ignore_nans=ignore_nans,
                **kwargs,
            )
        elif isinstance(divergence_loss, nn.Module):
            self.divergence_loss = divergence_loss
        else:
            raise ValueError(f"Invalid divergence loss: {divergence_loss}")

        self.register_buffer("divergence_loss_weight", torch.tensor(divergence_loss_weight))

    def forward(
        self, 
        pred: Tensor, 
        target: Tensor, 
        squash: bool | tuple = True,
        scalar_indices: tuple[int, ...] | None = None,
        without_scalars: list[str] | list[int] | None = None,
        mwm_mask: Optional[Tensor] = None, 
        **kwargs
    ) -> Dict[str, Tensor]:
        """Compute the VAE loss combining reconstruction and KL divergence.

        Parameters
        ----------
        pred : Tensor
            Predictions tensor, shape (bs, (timesteps), lat*lon, n_outputs)
        target : Tensor
            Target tensor, shape (bs, (timesteps), lat*lon, n_outputs)
        squash : bool | tuple, optional
            Whether to reduce/sum the loss over dimensions, by default True.
            If tuple, specifies dimensions to reduce over.
        scalar_indices : tuple[int, ...] | None, optional
            Indices to use for scaling, by default None
        without_scalars : list[str] | list[int] | None, optional
            List of scalars to exclude, by default None
        mwm_mask : Optional[Tensor], optional
            Mask tensor for masked weather modeling, by default None.
            Shape (bs, timesteps, ens, 1, lat*lon, n_outputs), where 1 indicates 
            masked points that the model needs to predict.

        Returns
        -------
        Dict[str, Tensor]
            Dictionary containing the total VAE loss and its components
        """
        x_rec = pred
        x_target = target

        # Get latent space parameters
        z_mu = kwargs.pop("x_latent")
        z_logvar = kwargs.pop("x_latent_logvar")

        # Compute divergence loss
        div_loss = self.divergence_loss(
            z_mu, 
            z_logvar, 
            squash=squash, 
            feature_scale=False, 
            feature_indices=kwargs.get("feature_indices", None)
        )

        # Scale the divergence loss by feature dimension
        # The reconstruction loss is scaled by feature weights during training with fw_i/sum(fw_i),
        # so on average each idx gets a scaling of 1/n (where n is feature count)
        if kwargs.get("feature_scale", True):
            div_loss = div_loss / self.reconstruction_loss.feature_weights.numel()

        # Compute reconstruction loss with or without mask
        if mwm_mask is None:
            rec_loss = self.reconstruction_loss(
                x_rec, 
                x_target, 
                squash=squash, 
                feature_scale=kwargs.get("feature_scale", True), 
                feature_indices=kwargs.get("feature_indices", None),
                scalar_indices=scalar_indices,
                without_scalars=without_scalars
            )
        else:
            # Without squashing or feature scaling initially
            rec_loss = self.reconstruction_loss(
                x_rec, 
                x_target, 
                squash=False, 
                feature_scale=False, 
                feature_indices=kwargs.get("feature_indices", None),
                scalar_indices=scalar_indices,
                without_scalars=without_scalars
            )

            # Apply mask-aware scaling
            if kwargs.get("feature_scale", True):
                # 1) Undo the area weighting that happens in the reconstruction losses
                rec_loss = rec_loss * (self.reconstruction_loss.node_weights.sum(dim=-2) / self.reconstruction_loss.node_weights)

                # 2) Calculate grid x feature scaling factor (set to 0 where mask is 0)
                scale_factor_grid_feature_numerator = self.reconstruction_loss.node_weights * self.reconstruction_loss.feature_weights
                scale_factor_grid_feature_numerator = torch.where(
                    mwm_mask == 1, 
                    scale_factor_grid_feature_numerator, 
                    torch.zeros_like(scale_factor_grid_feature_numerator)
                )

                # 3) Calculate the denominator as the sum where the mask is 1
                scale_factor_grid_feature_denominator = self.sum_function(scale_factor_grid_feature_numerator)

                # 4) Apply scaling factor
                rec_loss = rec_loss * (scale_factor_grid_feature_numerator / scale_factor_grid_feature_denominator)
            else:
                # Grid dimension reweighting only
                rec_loss = rec_loss * (self.reconstruction_loss.node_weights.sum(dim=-2) / self.reconstruction_loss.node_weights)

                # Calculate scaling factors based on mask
                scale_factor_grid_numerator = torch.where(
                    mwm_mask == 1,
                    self.reconstruction_loss.node_weights.expand_as(rec_loss), 
                    torch.zeros_like(rec_loss)
                )

                scale_factor_grid_denominator = self.sum_function(scale_factor_grid_numerator)

                # Apply scaling
                rec_loss = rec_loss * (scale_factor_grid_numerator / scale_factor_grid_denominator)

            # Apply squashing if requested
            if squash:
                rec_loss = rec_loss.sum(dim=squash if isinstance(squash, tuple) else (-5, -3, -2, -1))

        # Combine losses with weighting
        vae_loss = rec_loss.sum() + self.divergence_loss_weight * div_loss.sum()
        
        # Return loss components in a dictionary
        output = {
            self.name: vae_loss,
            f"{self.name_reconstruction}": rec_loss,
            f"{self.name_latent}": div_loss,
        }

        # Calculate adversarial loss if discriminator is provided
        if hasattr(self, 'discriminator_loss') and self.discriminator_loss is not None and "x_rec_adversarial" in kwargs:
            # Get adversarial output (features or classifications from a discriminator)
            x_rec_adversarial = kwargs["x_rec_adversarial"]
            adv_loss = self.forward_discriminator_loss(x_rec_adversarial, x_target)
            output[f"discriminator_{self.discriminator_loss.name}"] = adv_loss

        #     vae_loss = vae_loss + self.adversarial_loss_weight * adv_loss

        return output
    
    # def forward_discriminator_loss(self, x_rec_adversarial: Tensor, x_target_adversarial: Tensor):
    #     adv_loss = self.discriminator_loss(x_rec_adversarial, x_target_adversarial)

    #     return adv_loss

    @cached_property
    def name(self) -> str:
        """Return the name of the loss for logging.
        
        Returns
        -------
        str
            The combined name string for the VAE loss, including both
            reconstruction and latent components.
        """
        str_ = "bvae"
        str_ += f"_{self.name_reconstruction}"
        str_ += f"_{self.name_latent}"
        return str_

    @cached_property
    def name_reconstruction(self) -> str:
        """Return the name of the reconstruction loss component.
        
        Returns
        -------
        str
            The name of the reconstruction loss component.
        """
        return f"recon_{self.reconstruction_loss.name}"

    @cached_property
    def name_latent(self) -> str:
        """Return the name of the latent/divergence loss component.
        
        Returns
        -------
        str
            The name of the latent/divergence loss component.
        """
        return f"latent_{self.divergence_loss.name}"


class VQVAELossLucidRain(BaseWeightedLoss):
    """Vector Quantized VAE loss implementation based on the LucidRains VQ approach.
    
    Combines reconstruction loss with the commitment loss from the VQ codebook.
    """

    def __init__(
        self,
        node_weights: Tensor,
        feature_weights: Tensor,
        reconstruction_loss: dict | DictConfig | nn.Module,
        latent_node_weights: Tensor,
        commitment_weight: float = 0.25,
        ignore_nans: bool | None = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        node_weights : Tensor
            Weights for each spatial node in the grid, shape (lat*lon, 1)
        feature_weights : Tensor
            Weights for each feature, shape (n_features,)
        reconstruction_loss : dict | DictConfig | nn.Module
            Loss function for reconstruction term
        latent_node_weights : Tensor
            Weights for each spatial node in the latent space
        commitment_weight : float, optional
            Weight factor for the commitment loss, by default 0.25
        ignore_nans : bool | None, optional
            If True, use nanmean/nansum to ignore NaNs in loss computation, by default False
        """
        super().__init__(
            node_weights=node_weights,
            ignore_nans=ignore_nans,
            **kwargs,
        )
        
        self.commitment_weight = commitment_weight
        self.register_buffer("latent_node_weights", latent_node_weights[..., None], persistent=True)

        # Initialize reconstruction loss
        if isinstance(reconstruction_loss, (dict, DictConfig)):
            self.reconstruction_loss = instantiate_debug(
                reconstruction_loss,
                node_weights=node_weights,
                feature_weights=feature_weights,
                ignore_nans=ignore_nans,
                **kwargs,
            )
        elif isinstance(reconstruction_loss, nn.Module):
            self.reconstruction_loss = reconstruction_loss
        else:
            raise ValueError(f"Invalid reconstruction loss: {reconstruction_loss}")
        
    def forward(
        self, 
        pred: Tensor, 
        target: Tensor, 
        squash: Union[bool, tuple] = True,
        scalar_indices: tuple[int, ...] | None = None,
        without_scalars: list[str] | list[int] | None = None,
        mwm_mask: Optional[Tensor] = None,
        map_loss_breakdown: dict | None = None,
        **kwargs
    ) -> dict:
        """Compute the VQVAE loss.

        Parameters
        ----------
        pred : Tensor
            Predictions tensor, shape (bs, (timesteps), lat*lon, n_outputs)
        target : Tensor
            Target tensor, shape (bs, (timesteps), lat*lon, n_outputs)
        squash : bool | tuple, optional
            Whether to reduce/sum the loss over dimensions, by default True
        scalar_indices : tuple[int, ...] | None, optional
            Indices to use for scaling, by default None
        without_scalars : list[str] | list[int] | None, optional
            List of scalars to exclude, by default None
        mwm_mask : Optional[Tensor], optional
            Mask tensor, shape (bs, timesteps, ens, 1, lat*lon, n_outputs), by default None
            Used for masked weather modeling, where 1 indicates positions to predict
        map_loss_breakdown : dict | None, optional
            Dictionary containing VQ-specific loss components from the VQ layer, 
            including 'loss', by default None

        Returns
        -------
        dict
            Dictionary containing the total VQVAE loss and its components
        """
        x_rec = pred
        x_target = target 

        # Compute reconstruction loss with or without mask
        if mwm_mask is None:
            rec_loss = self.reconstruction_loss(
                x_rec, 
                x_target, 
                squash=squash, 
                feature_scale=kwargs.get("feature_scale", True), 
                feature_indices=kwargs.get("feature_indices", None),
                scalar_indices=scalar_indices,
                without_scalars=without_scalars
            )
        else:
            # Without squashing or feature scaling initially
            rec_loss = self.reconstruction_loss(
                x_rec, 
                x_target, 
                squash=False, 
                feature_scale=False, 
                feature_indices=kwargs.get("feature_indices", None),
                scalar_indices=scalar_indices,
                without_scalars=without_scalars
            )

            rec_loss = rec_loss * mwm_mask.numel() /  mwm_mask.sum()

        # Get VQ-specific loss components from map_loss_breakdown
        if map_loss_breakdown is None:
            raise ValueError("map_loss_breakdown cannot be None for VQVAELossLucidRain")
            
        vq_loss = map_loss_breakdown.pop("loss")

        # Calculate total loss if dimensions match or if fully squashed
        if rec_loss.numel() == vq_loss.numel() or squash is True:
            total_loss = rec_loss + vq_loss
        else:
            total_loss = rec_loss.sum() + vq_loss.sum()

        # Return loss components in a dictionary
        output = {
            self.name: total_loss,
            self.name_reconstruction: rec_loss,
            self.name_latent: vq_loss,
            **map_loss_breakdown,
        }
        
        return output
        
    @cached_property
    def name(self) -> str:
        """Return the name of the loss for logging.
        
        Returns
        -------
        str
            The combined name string for the VQVAE loss.
        """
        return f"vqvae_latentloss_{self.reconstruction_loss.name}"

    @cached_property
    def name_reconstruction(self) -> str:
        """Return the name of the reconstruction loss component.
        
        Returns
        -------
        str
            The name of the reconstruction loss component.
        """
        return f"recon_{self.reconstruction_loss.name}"

    @cached_property
    def name_latent(self) -> str:
        """Return the name of the latent loss component.
        
        Returns
        -------
        str
            The name of the Vector Quantization latent loss component.
        """
        return f"latent_loss"
