import logging
from abc import ABC
from abc import abstractmethod
from typing import Optional

import einops
import torch
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint

from anemoi.models.distributed.graph import gather_channels
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import change_channels_in_shape
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.mlp import MLP
from anemoi.models.layers.sparse_projector import build_sparse_projector
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class BaseNoiseInjector(nn.Module, ABC):
    """Abstract base class for noise injection strategies.

    Subclasses must implement the forward method which takes an input tensor
    and returns a tuple of (modified_tensor, noise_or_none).
    """

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        batch_size: int,
        ensemble_size: int,
        grid_size: int,
        shard_shapes_ref: tuple[tuple[int], tuple[int]],
        noise_dtype: torch.dtype = torch.float32,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Forward pass for noise injection.

        Parameters
        ----------
        x : Tensor
            Input tensor to potentially modify
        batch_size : int
            Batch size
        ensemble_size : int
            Ensemble size
        grid_size : int
            Grid size
        shard_shapes_ref : tuple[tuple[int], tuple[int]]
            Shard shapes when sharded
        noise_dtype : torch.dtype, optional
            Data type for noise tensor
        model_comm_group : ProcessGroup, optional
            Model communication group

        Returns
        -------
        tuple[Tensor, Optional[Tensor]]
            Tuple of (output_tensor, noise_tensor_or_none):
                - output_tensor: The (potentially) modified input tensor
                - noise_tensor_or_none: The noise tensor for conditioning,
                  or None if noise is injected directly into output_tensor
        """
        ...


class NoOpNoiseInjector(BaseNoiseInjector):
    """No-op noise injector that passes through input unchanged.

    Use this when noise injection is disabled.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize NoOpNoiseInjector."""
        super().__init__()

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        ensemble_size: int,
        grid_size: int,
        shard_shapes_ref: tuple[tuple[int], tuple[int]],
        noise_dtype: torch.dtype = torch.float32,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[Tensor, None]:
        """Pass through input unchanged with no noise."""
        return x, None


class NoiseConditioning(BaseNoiseInjector):
    """Noise Conditioning."""

    def __init__(
        self,
        *,
        noise_std: int,
        noise_channels_dim: int,
        noise_mlp_hidden_dim: int,
        layer_kernels: DotDict,
        noise_matrix: Optional[str] = None,
        transpose_noise_matrix: bool = False,
        row_normalize_noise_matrix: bool = False,
        autocast: bool = False,
        num_channels: Optional[int] = None,
    ) -> None:
        """Initialize NoiseConditioning."""
        super().__init__()
        assert noise_channels_dim > 0, "Noise channels must be a positive integer"
        assert noise_mlp_hidden_dim > 0, "Noise channels must be a positive integer"

        self.noise_std = noise_std

        # Noise channels
        self.noise_channels = noise_channels_dim

        self.layer_factory = load_layer_kernels(layer_kernels)

        self.noise_mlp = MLP(
            noise_channels_dim,
            noise_mlp_hidden_dim,
            noise_channels_dim,
            layer_kernels=self.layer_factory,
            n_extra_layers=-1,
            final_activation=False,
            layer_norm=True,
        )

        self.noise_projector = None
        if noise_matrix is not None:
            self.noise_projector = build_sparse_projector(
                file_path=noise_matrix,
                transpose=transpose_noise_matrix,
                row_normalize=row_normalize_noise_matrix,
                autocast=autocast,
            )
            LOGGER.info("Noise projector matrix shape = %s", self.noise_projector.projection_matrix.shape)

        LOGGER.info("processor noise channels = %d", self.noise_channels)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        ensemble_size: int,
        grid_size: int,
        shard_shapes_ref: tuple[tuple[int], tuple[int]],
        noise_dtype: torch.dtype = torch.float32,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[Tensor, Tensor]:

        noise_shape = (
            batch_size,
            ensemble_size,
            grid_size if self.noise_projector is None else self.noise_projector.projection_matrix.shape[1],
            self.noise_channels,
        )

        noise = torch.randn(size=noise_shape, dtype=noise_dtype, device=x.device) * self.noise_std
        noise.requires_grad = False

        noise_shard_shapes_final = change_channels_in_shape(shard_shapes_ref, self.noise_channels)

        if self.noise_projector is not None:
            noise_shard_shapes = get_shard_shapes(noise, -1, model_comm_group)
            noise = shard_tensor(noise, -1, noise_shard_shapes, model_comm_group)  # split across channels

            noise = einops.rearrange(
                noise, "batch ensemble grid vars -> (batch ensemble) grid vars"
            )  # batch and ensemble always 1 when sharded

            noise = self.noise_projector(noise)  # to shape of hidden grid

            noise = einops.rearrange(noise, "bse grid vars -> (bse grid) vars")  # shape of x
            noise = gather_channels(
                noise, noise_shard_shapes_final, model_comm_group
            )  # sharded grid dim, full channels
        else:
            noise = einops.rearrange(noise, "batch ensemble grid vars -> (batch ensemble grid) vars")  # shape of x
            noise_shard_shapes = get_shard_shapes(noise, 0, model_comm_group)
            noise = shard_tensor(noise, 0, noise_shard_shapes, model_comm_group)  # sharded grid dim, full channels

        noise = checkpoint(self.noise_mlp, noise, use_reentrant=False)

        LOGGER.debug("Noise noise.shape = %s, noise.norm: %.9e", noise.shape, torch.linalg.norm(noise))

        return x, noise


class NoiseInjector(BaseNoiseInjector):
    """Noise Injection Module.

    Generates noise and projects it directly into the input tensor,
    returning None for the noise (since it's already incorporated).
    """

    def __init__(
        self,
        *,
        noise_std: int,
        noise_channels_dim: int,
        noise_mlp_hidden_dim: int,
        num_channels: int,
        layer_kernels: DotDict,
        noise_matrix: Optional[str] = None,
    ) -> None:
        """Initialize NoiseInjector.

        Parameters
        ----------
        noise_std : int
            Standard deviation for noise generation
        noise_channels_dim : int
            Number of noise channels
        noise_mlp_hidden_dim : int
            Hidden dimension of noise MLP
        num_channels : int
            Number of model channels for projection
        layer_kernels : DotDict
            Layer kernel configurations
        noise_matrix : str, optional
            Optional path to noise truncation matrix
        """
        super().__init__()

        self._noise_conditioning = NoiseConditioning(
            noise_std=noise_std,
            noise_channels_dim=noise_channels_dim,
            noise_mlp_hidden_dim=noise_mlp_hidden_dim,
            layer_kernels=layer_kernels,
            noise_matrix=noise_matrix,
        )
        self.noise_channels = noise_channels_dim
        self.projection = nn.Linear(num_channels + self.noise_channels, num_channels)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        ensemble_size: int,
        grid_size: int,
        shard_shapes_ref: tuple[tuple[int], tuple[int]],
        noise_dtype: torch.dtype = torch.float32,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[Tensor, None]:
        """Generate noise and inject it into the input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor to modify
        batch_size : int
            Batch size
        ensemble_size : int
            Ensemble size
        grid_size : int
            Grid size
        shard_shapes_ref : tuple[tuple[int], tuple[int]]
            Shard shapes when sharded
        noise_dtype : torch.dtype, optional
            Data type for noise tensor
        model_comm_group : ProcessGroup, optional
            Model communication group

        Returns
        -------
        tuple[Tensor, None]
            Tuple of (modified_x, None): Modified tensor with noise injected
        """
        x, noise = self._noise_conditioning(
            x=x,
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            grid_size=grid_size,
            shard_shapes_ref=shard_shapes_ref,
            noise_dtype=noise_dtype,
            model_comm_group=model_comm_group,
        )

        return (
            self.projection(torch.cat([x, noise], dim=-1)),
            None,
        )
