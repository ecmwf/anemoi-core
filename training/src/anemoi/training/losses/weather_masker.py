import torch
import numpy as np
from typing import List, Dict, Union, Tuple, Optional
from dataclasses import dataclass
from torch import nn
from anemoi.graphs.create import GraphCreator
from anemoi.graphs.edges.builder import KNNEdges
from torch_geometric.data import HeteroData
from collections import defaultdict
from omegaconf import DictConfig
from anemoi.models.data_indices.collection import IndexCollection
import einops
import random
from collections import OrderedDict
from anemoi.training.utils.buffer_dict import BufferDict
from collections.abc import Sequence
import logging

LOGGER = logging.getLogger(__name__)

class WeatherMaskGeneratorRandom(nn.Module):
    """
    Generates masks for weather modeling with random masking patterns.
    
    In the final masks:
        0 values indicate masked/hidden points that the model needs to predict
        1 values indicate unmasked points that the model can see
    """
    def __init__(
        self,
        feature_in_size: int,
        feature_out_size: int,
        feature_mapping: dict[str, dict[str, int]],
        mask_ratio: float | list[float] = 0.35,
        **kwargs,
    ):
        """
        Initialize the mask generator with variable patch sizes.
        
        Parameters
        ----------
        feature_in_size : int
            Number of input features
        feature_out_size : int
            Number of output features
        feature_mapping : dict[str, dict[str, int]]
            Mapping from output feature names to input feature indices
        mask_ratio : float | list[float], optional
            Proportion of data to mask. If a list of two values is provided,
            the mask ratio will be randomly sampled between these values for each batch.
            By default 0.35
        """
        super().__init__()

        self.feature_in_size = feature_in_size
        self.feature_out_size = feature_out_size

        self.mask_ratio = mask_ratio
        if isinstance(mask_ratio, list):
            assert len(mask_ratio) == 2, "mask_ratio must be a list of two values, the minimum and maximum mask ratio"
            
        # Convert feature mapping to tensor
        li_feature_idx_outp_to_inp = [
            v['model_input_idx'] for k, v in sorted(feature_mapping.items(), key=lambda x: feature_mapping[x[0]]['model_output_idx'])
        ]
        self.register_buffer("li_feature_idx_outp_to_inp", torch.as_tensor(li_feature_idx_outp_to_inp))
        
    def generate_masks(
        self,
        x_target: torch.Tensor,
        x_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate masks using random values based on the mask ratio.
        
        Parameters
        ----------
        x_target : torch.Tensor
            Target tensor with shape (batch_size, time, ensemble, grid, n_features)
        x_input : torch.Tensor
            Input tensor with shape (batch_size, time, ensemble, grid, n_features)
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of input and target masks.
        """
        batch_size, time, ensemble, grid, _ = x_target.shape
        device = x_target.device

        # Generate random mask values
        random_values = torch.rand((batch_size, time, ensemble, grid, self.feature_out_size), device=device)

        # Create target mask based on mask ratio
        if isinstance(self.mask_ratio, float):
            target_mask = torch.where(random_values < self.mask_ratio, 
                                       torch.ones_like(random_values), 
                                       torch.zeros_like(random_values))
        else:
            # Sample a different mask ratio for each element in the batch
            mask_ratio = torch.rand(x_target.shape, device=device) * (self.mask_ratio[1] - self.mask_ratio[0]) + self.mask_ratio[0]
            target_mask = torch.where(random_values < mask_ratio, 
                                      torch.ones_like(random_values), 
                                      torch.zeros_like(random_values))

        # Create input mask using feature mapping
        input_mask = torch.zeros_like(x_input)
        
        # Map the target mask to the corresponding input features
        # NOTE: Forcing parameters (those without a corresponding output) remain unmasked
        input_mask[..., self.li_feature_idx_outp_to_inp] = target_mask

        return input_mask, target_mask
    
class WeatherMaskGeneratorTube(nn.Module):
    
    pass
