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

import torch
from torch import Tensor
from torch import nn

from anemoi.models.layers.geocyclic import GeocyclicDepthwiseConv
from anemoi.models.layers.layout import LatLonLayout
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


def init_branch(*layers: nn.Module, last_scale: float = 0.1) -> None:
    """Initialise the layers of one branch, given in the order they are applied.

    Weights are drawn with Kaiming-normal scaling (fan-in, ReLU gain) and biases start at zero.
    The last layer's weights are scaled down so that the branch starts with a small output.
    """
    for i, layer in enumerate(layers):
        nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        if i == len(layers) - 1:
            with torch.no_grad():
                layer.weight.mul_(last_scale)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class SpatialMixer(nn.Module):
    """A stencil per channel followed by a linear mix of the channels at each point."""

    def __init__(self, in_channels: int, out_channels: int, layer_kernels: DotDict, kernel_size: int = 5) -> None:
        super().__init__()
        self.stencil = GeocyclicDepthwiseConv(in_channels, kernel_size)
        self.mix = layer_kernels.Linear(in_channels, out_channels)

    def forward(self, x: Tensor, layout: LatLonLayout) -> Tensor:
        return self.mix(self.stencil(x, layout))


class LowRankBias(nn.Module):
    """A learned field over the globe, built from a few smooth patterns so that it needs few parameters.

    A small number of base maps is formed as sums of products of a latitude profile and a
    longitude profile, then mixed linearly into the output channels. The profiles start very
    small, so the field starts close to zero.
    """

    def __init__(
        self,
        num_channels: int,
        nlat: int,
        nlon: int,
        layer_kernels: DotDict,
        rank: int = 128,
        num_base_maps: int = 8,
    ) -> None:
        super().__init__()
        self.coefficients = nn.Parameter(torch.randn(num_base_maps, rank) * 1e-3)
        self.lat_profiles = nn.Parameter(torch.randn(rank, nlat) * 1e-3)
        self.lon_profiles = nn.Parameter(torch.randn(rank, nlon) * 1e-3)
        self.projection = layer_kernels.Linear(num_base_maps, num_channels, bias=False)

    def forward(self, x: Tensor, layout: LatLonLayout) -> Tensor:
        """Add the field to x, of shape (batch * local nodes, channels)."""
        base_maps = torch.einsum("ck,ki,kj->ijc", self.coefficients, self.lat_profiles, self.lon_profiles)
        start, end = layout.local_nodes
        base_maps = base_maps.reshape(-1, base_maps.shape[-1])
        if layout.grid_index is None:
            base_maps = base_maps[start:end]
        else:
            base_maps = base_maps.index_select(0, layout.grid_index[start:end])
        field = self.projection(base_maps)
        return (x.view(layout.batch_size, end - start, -1) + field).view(x.shape)
