import logging
from typing import Optional

import einops
import scipy.sparse
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
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.models.truncation import make_truncation_matrix
from anemoi.models.truncation import truncate_fields
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class NoiseConditioning(nn.Module):
    """Noise Conditioning."""

    def __init__(
        self,
        *,
        noise_std: int,
        noise_channels_dim: int,
        noise_mlp_hidden_dim: int,
        layer_kernels: DotDict,
        noise_matrix: Optional[str] = None,
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

        self.A_noise = None
        if noise_matrix is not None:
            interpolation_data = scipy.sparse.load_npz(noise_matrix)
            self.A_noise = make_truncation_matrix(interpolation_data)
            LOGGER.info("Loaded noise matrix from %s with shape %s", noise_matrix, self.A_noise.shape)

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
            grid_size if self.A_noise is None else self.A_noise.shape[1],
            self.noise_channels,
        )

        noise = torch.randn(size=noise_shape, dtype=noise_dtype, device=x.device) * self.noise_std
        noise.requires_grad = False

        noise_shard_shapes_final = change_channels_in_shape(shard_shapes_ref, self.noise_channels)

        if self.A_noise is not None:
            self.A_noise = self.A_noise.to(noise.device)

            noise_shard_shapes = get_shard_shapes(noise, -1, model_comm_group)
            noise = shard_tensor(noise, -1, noise_shard_shapes, model_comm_group)  # split across channels

            noise = einops.rearrange(
                noise, "batch ensemble grid vars -> (batch ensemble) grid vars"
            )  # batch and ensemble always 1 when sharded
            noise = truncate_fields(noise, self.A_noise)  # to shape of hidden grid
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


class NoiseInjector(NoiseConditioning):
    """Noise Injection Module."""

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
        """Initialize NoiseInjector."""
        super().__init__(
            noise_std=noise_std,
            noise_channels_dim=noise_channels_dim,
            noise_mlp_hidden_dim=noise_mlp_hidden_dim,
            layer_kernels=layer_kernels,
            noise_matrix=noise_matrix,
        )
        self.projection = nn.Linear(num_channels + self.noise_channels, num_channels)  # Fold noise into the channels

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

        x, noise = super().forward(
            x=x,
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            grid_size=grid_size,
            shard_shapes_ref=shard_shapes_ref,
            noise_dtype=noise_dtype,
            model_comm_group=model_comm_group,
        )

        return (
            self.projection(
                torch.cat(
                    [x, noise],
                    dim=-1,  # feature dimension
                ),
            ),
            None,
        )
