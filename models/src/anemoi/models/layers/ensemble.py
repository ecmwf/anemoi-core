import logging
from typing import Optional

import torch
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import change_channels_in_shape
from anemoi.models.layers.mlp import MLP
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class NoiseInjector(nn.Module):
    """Noise Injection Module."""

    def __init__(
        self,
        *,
        noise_std: int,
        noise_channels_dim: int,
        noise_mlp_hidden_dim: int,
        num_channels: int,
        layer_kernels: DotDict,
        inject_noise: bool = True,
    ) -> None:
        """Initialize NoiseInjector."""
        super().__init__()
        assert noise_channels_dim > 0, "Noise channels must be a positive integer"
        assert noise_mlp_hidden_dim > 0, "Noise channels must be a positive integer"

        # Switch off noise injection
        self.inject_noise = inject_noise
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

        self.projection = nn.Linear(num_channels + self.noise_channels, num_channels)  # Fold noise into the channels

        LOGGER.debug("processor noise channels = %d", self.noise_channels)

    def make_noise(self, noise_ref):
        tensor_shape = (*noise_ref.shape[:-1], self.noise_channels)
        noise = torch.randn(size=tensor_shape, dtype=noise_ref.dtype, device=noise_ref.device) * self.noise_std
        noise.requires_grad = False
        return noise

    def forward(
        self,
        x: Tensor,
        noise_ref: Tensor,
        shard_shapes: list[list[int], list[int]],
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[Tensor, Tensor]:

        noise = self.make_noise(noise_ref)

        shapes_sharding = change_channels_in_shape(shard_shapes, self.noise_channels)
        noise = shard_tensor(noise, 0, shapes_sharding, model_comm_group)
        noise = checkpoint(self.noise_mlp, noise, use_reentrant=False)

        LOGGER.debug("Noise noise.shape = %s, noise.norm: %.9e", noise.shape, torch.linalg.norm(noise))

        return (
            self.projection(
                torch.cat(
                    [x, noise],
                    dim=-1,  # feature dimension
                ),
            ),
            None,
        )


class NoiseConditioning(NoiseInjector):
    """Noise Conditioning."""

    def __init__(
        self,
        *,
        noise_std: int,
        noise_channels_dim: int,
        noise_mlp_hidden_dim: int,
        num_channels: int,
        layer_kernels: DotDict,
        inject_noise: bool = True,
    ) -> None:
        """Initialize NoiseConditioning."""
        super().__init__(
            noise_std=noise_std,
            noise_channels_dim=noise_channels_dim,
            noise_mlp_hidden_dim=noise_mlp_hidden_dim,
            layer_kernels=layer_kernels,
            num_channels=num_channels,
            inject_noise=inject_noise,
        )
        self.projection = None

    def forward(
        self,
        x: Tensor,
        noise_ref: Tensor,
        shard_shapes: list[list[int], list[int]],
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[Tensor, Tensor]:

        noise = self.make_noise(noise_ref)

        shapes_sharding = change_channels_in_shape(shard_shapes, self.noise_channels)
        noise = shard_tensor(noise, 0, shapes_sharding, model_comm_group)
        noise = checkpoint(self.noise_mlp, noise, use_reentrant=False)

        return x, noise
