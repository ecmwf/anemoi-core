# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Steps of the advection-diffusion-reaction block (PARADIS, arXiv:2601.21151).

Each step is built from its own config by ``ADRProcessorBlock``, which passes in the grid:
``num_channels``, ``nlat``, ``nlon``, ``grid_index`` and ``layer_kernels``, and ``time_step`` for
the advection step. A step returns the increment that the block adds to the state.
"""

import logging
from typing import Optional

import torch
from torch import Tensor
from torch import nn

from anemoi.models.layers.semi_lagrangian import LatLonLayout
from anemoi.models.layers.semi_lagrangian import LowRankBias
from anemoi.models.layers.semi_lagrangian import SemiLagrangianAdvection
from anemoi.models.layers.semi_lagrangian import SpatialMixer
from anemoi.models.layers.semi_lagrangian import init_branch
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class NeuralSemiLagrangian(nn.Module):
    """Neural Semi-Lagrangian module.

    The state moves towards its advected version by a learned share per channel.

    Attributes
    ----------
    transport : SemiLagrangianAdvection
        The semi-Lagrangian advection module.
    share : nn.Parameter
        The learned share per channel, passed through a sigmoid. It determines how much of the
        advected state is applied to the current state. 
    """

    def __init__(
        self,
        *,
        num_channels: int,
        nlat: int,
        nlon: int,
        time_step: float,
        layer_kernels: DotDict,
        grid_index: Optional[Tensor] = None,
        advection_channels: int = 768,
        num_heads: Optional[int] = None,
        velocity_hidden_dim: int = 384,
        kernel_size: int = 5,
        interpolation: str = "bicubic",
        bias_rank: int = 128,
        bias_base_maps: int = 8,
        cartesian_displacement: bool = False,
    ) -> None:
        super().__init__()
        if num_heads is None:
            LOGGER.debug(
                f"{self.__class__.__name__}: num_heads not specified, using {advection_channels} (advection channels)"
            )
            num_heads = advection_channels

        self.transport = SemiLagrangianAdvection(
            num_channels=num_channels,
            advection_channels=advection_channels,
            num_heads=num_heads,
            velocity_hidden_dim=velocity_hidden_dim,
            nlat=nlat,
            nlon=nlon,
            time_step=time_step,
            layer_kernels=layer_kernels,
            kernel_size=kernel_size,
            interpolation=interpolation,
            bias_rank=bias_rank,
            bias_base_maps=bias_base_maps,
            cartesian_displacement=cartesian_displacement,
            grid_index=grid_index,
        )
        # This determines how much of the advected state is applied to the current state.
        self.share = nn.Parameter(torch.full((num_channels,), -1.0))

    def forward(self, x: Tensor, layout: LatLonLayout, cond: Optional[Tensor] = None) -> Tensor:
        share = torch.sigmoid(self.share).to(x.dtype)
        return share * (self.transport(x, layout, cond) - x)


class Diffusion(nn.Module):
    """Diffusion module."""

    def __init__(
        self,
        *,
        num_channels: int,
        nlat: int,
        nlon: int,
        layer_kernels: DotDict,
        kernel_size: int = 5,
        layer_norm: bool = True,
        low_rank_bias: bool = True,
        bias_rank: int = 128,
        bias_base_maps: int = 8,
        **kwargs,
    ) -> None:
        del kwargs
        super().__init__()
        self.layer_norm = layer_kernels.LayerNorm(normalized_shape=num_channels) if layer_norm else None
        self.mixer = SpatialMixer(num_channels, num_channels, layer_kernels, kernel_size)
        self.bias = (
            LowRankBias(num_channels, nlat, nlon, layer_kernels, rank=bias_rank, num_base_maps=bias_base_maps)
            if low_rank_bias
            else None
        )
        init_branch(self.mixer.stencil.conv, self.mixer.mix)

    def forward(self, x: Tensor, layout: LatLonLayout, cond: Optional[Tensor] = None) -> Tensor:
        """Forward pass of the diffusion module.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, num_channels, nlat, nlon).
        layout : LatLonLayout
            Layout object describing the latitude and longitude grid.
        cond : Tensor, optional
            Conditioning tensor, by default None.

        Returns
        -------
        Tensor
            Output tensor of the same shape as the input.
        """
        if self.layer_norm is not None:
            x = self.layer_norm(x, **({"cond": cond} if cond is not None else {}))
        x = self.mixer(x, layout)
        if self.bias is not None:
            x = self.bias(x, layout)
        return x


class Reaction(nn.Module):
    """Reaction module."""

    def __init__(
        self,
        *,
        num_channels: int,
        nlat: int,
        nlon: int,
        layer_kernels: DotDict,
        hidden_dim: int = 896,
        num_layers: int = 4,
        layer_norm: bool = True,
        low_rank_bias: bool = True,
        bias_rank: int = 128,
        bias_base_maps: int = 8,
        **kwargs,
    ) -> None:
        del kwargs
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be at least 2, got {num_layers}.")
        self.layer_norm = layer_kernels.LayerNorm(normalized_shape=num_channels) if layer_norm else None
        dims = [num_channels] + [hidden_dim] * (num_layers - 1) + [num_channels]
        self.layers = nn.ModuleList(layer_kernels.Linear(dims[i], dims[i + 1]) for i in range(num_layers))
        self.bias = (
            LowRankBias(hidden_dim, nlat, nlon, layer_kernels, rank=bias_rank, num_base_maps=bias_base_maps)
            if low_rank_bias
            else None
        )
        self.activation = layer_kernels.Activation()
        init_branch(*self.layers)

    def forward(self, x: Tensor, layout: LatLonLayout, cond: Optional[Tensor] = None) -> Tensor:
        """Forward pass of the reaction module.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, num_channels, nlat, nlon).
        layout : LatLonLayout
            Layout object describing the latitude and longitude grid.
        cond : Tensor, optional
            Conditioning tensor, by default None.

        Returns
        -------
        Tensor
            Output tensor of the same shape as the input.
        """
        if self.layer_norm is not None:
            x = self.layer_norm(x, **({"cond": cond} if cond is not None else {}))
        x = self.layers[0](x)
        if self.bias is not None:
            x = self.bias(x, layout)
        for layer in self.layers[1:]:
            x = layer(self.activation(x))
        return x
