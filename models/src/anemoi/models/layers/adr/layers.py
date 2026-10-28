# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Spatial parts of the ADR steps, one per geometry.

Each geometry has a spatial mixer, which looks at the neighbours of each point and mixes the
channels, and a low-rank bias, a learned field over the globe:

- on a regular latitude-longitude grid (``LatLonLayout``): ``LatLonSpatialMixer`` and
  ``LatLonLowRankBias``, on tensors of shape (batch, local rows, nlon, channels);
- on the nodes of a graph (``GraphLayout``): ``GraphSpatialMixer`` and ``GraphLowRankBias``, on
  tensors of shape (batch * local nodes, channels).

Mixers are called as ``mixer(x, layout, cond)`` and biases as ``bias(x, layout)``.
"""

import logging
from typing import Optional

import torch
from torch import Tensor
from torch import nn

from anemoi.models.layers.block import GraphConvProcessorBlock
from anemoi.models.layers.block import GraphTransformerProcessorBlock
from anemoi.models.layers.edge_stencil import EdgeDepthwiseConv
from anemoi.models.layers.geocyclic import GeocyclicDepthwiseConv
from anemoi.models.layers.layout import GraphLayout
from anemoi.models.layers.layout import LatLonLayout
from anemoi.models.layers.utils import compute_mlp_hidden_dim
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


class LatLonSpatialMixer(nn.Module):
    """A stencil per channel followed by a linear mix of the channels at each point."""

    def __init__(self, in_channels: int, out_channels: int, layer_kernels: DotDict, kernel_size: int = 5) -> None:
        super().__init__()
        self.stencil = GeocyclicDepthwiseConv(in_channels, kernel_size)
        self.mix = layer_kernels.Linear(in_channels, out_channels)

    def forward(self, x: Tensor, layout: LatLonLayout, cond: Optional[Tensor] = None) -> Tensor:
        return self.mix(self.stencil(x, layout))

    def branch_layers(self) -> tuple[nn.Module, ...]:
        """Layers that ``init_branch`` initialises, in the order they are applied."""
        return (self.stencil.conv, self.mix)


class GraphSpatialMixer(nn.Module):
    """A graph block over the hidden-to-hidden edges followed by a linear mix of the channels at each node.

    The graph block takes the place of the stencil of ``LatLonSpatialMixer``:

    - ``"edge_conv"``: an ``EdgeDepthwiseConv``, one learned weight per edge and channel. The
      closest analogue of the stencil on a grid, and by far the cheapest of the three;
    - ``"transformer"``: a ``GraphTransformerProcessorBlock``, attention over the edges;
    - ``"gnn"``: a ``GraphConvProcessorBlock``, message passing over the edges.

    All keep the number of channels; the transformer and the GNN also add their output to their
    input and bring their own MLP and layer norms. ``block_kwargs`` go to the block, e.g.
    ``graph_attention_backend`` or ``qk_norm`` for the transformer. Only the transformer passes the
    conditioning on to its layer norms; the other two ignore it, as the mixers on a grid do.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        layer_kernels: DotDict,
        block: str = "edge_conv",
        num_heads: int = 16,
        mlp_hidden_ratio: float = 1.0,
        **block_kwargs,
    ) -> None:
        super().__init__()
        if block == "edge_conv":
            self.block = EdgeDepthwiseConv(in_channels, edge_dim, layer_kernels, **block_kwargs)
        elif block == "transformer":
            self.block = GraphTransformerProcessorBlock(
                in_channels=in_channels,
                hidden_dim=compute_mlp_hidden_dim(in_channels, mlp_hidden_ratio),
                out_channels=in_channels,
                num_heads=num_heads,
                edge_dim=edge_dim,
                layer_kernels=layer_kernels,
                shard_strategy="edges",
                **block_kwargs,
            )
        elif block == "gnn":
            self.block = GraphConvProcessorBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                num_chunks=1,
                mlp_hidden_ratio=mlp_hidden_ratio,
                layer_kernels=layer_kernels,
                edge_dim=edge_dim,
                **block_kwargs,
            )
        else:
            raise ValueError(f"block must be 'edge_conv', 'transformer' or 'gnn', got {block}.")
        self.block_type = block
        self.mix = layer_kernels.Linear(in_channels, out_channels)

    def forward(self, x: Tensor, layout: GraphLayout, cond: Optional[Tensor] = None) -> Tensor:
        if self.block_type == "edge_conv":
            return self.mix(self.block(x, layout))
        if self.block_type == "transformer":
            x, _ = self.block(
                x,
                layout.edge_attr,
                layout.edge_index,
                layout.shard_info,
                layout.batch_size,
                layout.size,
                model_comm_group=layout.model_comm_group,
                cond=cond,
                halo_info=layout.halo_info,
            )
        else:
            x, _ = self.block(x, layout.edge_attr, layout.edge_index, layout.shard_info, layout.model_comm_group)
        return self.mix(x)

    def branch_layers(self) -> tuple[nn.Module, ...]:
        """Layers that ``init_branch`` initialises; the transformer and GNN blocks keep their own."""
        if self.block_type == "edge_conv":
            return (self.block.weight, self.mix)
        return (self.mix,)


class LatLonLowRankBias(nn.Module):
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
        """Add the field to x, of shape (batch, local rows, nlon, channels)."""
        start, end = layout.local_rows
        base_maps = torch.einsum("ck,ki,kj->ijc", self.coefficients, self.lat_profiles[:, start:end], self.lon_profiles)
        return x + self.projection(base_maps)


class GraphLowRankBias(nn.Module):
    """A learned field over the globe, like ``LatLonLowRankBias``, evaluated at the nodes' coordinates.

    The base maps are again sums of products of a latitude profile and a longitude profile, but
    the profiles are short series rather than a value per grid row and column, so they can be
    evaluated anywhere:

    - latitude: Chebyshev polynomials of sin(latitude) up to ``lat_degree``, that is
      cos(l * colatitude), which are smooth at the poles;
    - longitude: a Fourier series up to wavenumber ``lon_wavenumber``, wavenumber m tapered by
      cos(latitude)^m, as in spherical harmonics, so that the field has a single value at a pole
      and short rings near the poles see only low wavenumbers.

    The profiles start very small, so the field starts close to zero.
    """

    def __init__(
        self,
        num_channels: int,
        layer_kernels: DotDict,
        rank: int = 128,
        num_base_maps: int = 8,
        lat_degree: int = 32,
        lon_wavenumber: int = 32,
    ) -> None:
        super().__init__()
        self.coefficients = nn.Parameter(torch.randn(num_base_maps, rank) * 1e-3)
        self.lat_profiles = nn.Parameter(torch.randn(rank, lat_degree + 1) * 1e-3)
        self.lon_profiles = nn.Parameter(torch.randn(rank, 2 * lon_wavenumber + 1) * 1e-3)
        self.projection = layer_kernels.Linear(num_base_maps, num_channels, bias=False)
        self.lat_degree = lat_degree
        self.lon_wavenumber = lon_wavenumber

    def features(self, lat: Tensor, lon: Tensor) -> tuple[Tensor, Tensor]:
        """Latitude features (nodes, lat_degree + 1) and longitude features (nodes, 2 * lon_wavenumber + 1)."""
        degree = torch.arange(self.lat_degree + 1, device=lat.device, dtype=lat.dtype)
        lat_features = torch.cos(degree * (torch.pi / 2 - lat[:, None]))

        wavenumber = torch.arange(1, self.lon_wavenumber + 1, device=lon.device, dtype=lon.dtype)
        taper = lat.cos().clamp_min(0)[:, None] ** wavenumber
        angle = wavenumber * lon[:, None]
        lon_features = torch.cat([torch.ones_like(lon[:, None]), taper * angle.cos(), taper * angle.sin()], dim=-1)
        return lat_features, lon_features

    def forward(self, x: Tensor, layout: GraphLayout) -> Tensor:
        """Add the field to x, of shape (batch * local nodes, channels)."""
        start, end = layout.local_nodes
        lat_features, lon_features = self.features(layout.lat[start:end], layout.lon[start:end])
        profiles = (lat_features @ self.lat_profiles.T) * (lon_features @ self.lon_profiles.T)
        field = self.projection(profiles @ self.coefficients.T)
        return (x.view(layout.batch_size, end - start, -1) + field).view(x.shape)
