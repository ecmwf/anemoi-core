# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Callable
from typing import Optional
from typing import Union

from torch import Tensor
from torch import nn

from anemoi.models.layers.adr.layers import GraphLowRankBias
from anemoi.models.layers.adr.layers import GraphSpatialMixer
from anemoi.models.layers.adr.layers import LatLonLowRankBias
from anemoi.models.layers.adr.layers import LatLonSpatialMixer
from anemoi.models.layers.adr.layers import init_branch
from anemoi.models.layers.layout import GraphLayout
from anemoi.models.layers.layout import LatLonLayout
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class Diffusion(nn.Module):
    """Diffusion step: layer norm, a spatial mixer and a learned bias field.

    The geometry-specific subclasses (``LatLonDiffusion``, ``GraphDiffusion``) give the mixer and
    the bias, as ``mixer(in_channels, out_channels)`` and ``bias(num_channels)``.
    """

    def __init__(
        self,
        *,
        num_channels: int,
        layer_kernels: DotDict,
        mixer: Callable[[int, int], nn.Module],
        bias: Callable[[int], nn.Module],
        layer_norm: bool = True,
        low_rank_bias: bool = True,
    ) -> None:
        super().__init__()
        self.layer_norm = layer_kernels.LayerNorm(normalized_shape=num_channels) if layer_norm else None
        self.mixer = mixer(num_channels, num_channels)
        self.bias = bias(num_channels) if low_rank_bias else None
        init_branch(*self.mixer.branch_layers())

    def forward(self, x: Tensor, layout: Union[LatLonLayout, GraphLayout], cond: Optional[Tensor] = None) -> Tensor:
        """Forward pass of the diffusion step.

        Parameters
        ----------
        x : Tensor
            The state, of shape (batch, local rows, nlon, num_channels) on a latitude-longitude
            grid, or (batch * local nodes, num_channels) on a graph.
        layout : LatLonLayout or GraphLayout
            Layout of the state, matching the subclass.
        cond : Tensor, optional
            Conditioning tensor, shaped like x with the condition channels, by default None.

        Returns
        -------
        Tensor
            Output tensor of the same shape as the input.
        """
        cond_kwargs = {"cond": cond} if cond is not None else {}
        if self.layer_norm is not None:
            x = self.layer_norm(x, **cond_kwargs)
        x = self.mixer(x, layout, cond)
        if self.bias is not None:
            x = self.bias(x, layout)
        return x


class LatLonDiffusion(Diffusion):
    """Diffusion step on a regular latitude-longitude grid: a stencil per channel and a channel mix."""

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
        super().__init__(
            num_channels=num_channels,
            layer_kernels=layer_kernels,
            mixer=lambda i, o: LatLonSpatialMixer(i, o, layer_kernels, kernel_size),
            bias=lambda c: LatLonLowRankBias(
                c, nlat, nlon, layer_kernels, rank=bias_rank, num_base_maps=bias_base_maps
            ),
            layer_norm=layer_norm,
            low_rank_bias=low_rank_bias,
        )


class GraphDiffusion(Diffusion):
    """Diffusion step on the nodes of a graph: a graph block over the hidden-to-hidden edges and a channel mix.

    ``mixer_block`` is ``"edge_conv"``, ``"transformer"`` or ``"gnn"`` (see
    ``GraphSpatialMixer``), and ``mixer_kwargs`` go to the block (e.g. ``num_heads``,
    ``mlp_hidden_ratio``, ``graph_attention_backend``).
    """

    def __init__(
        self,
        *,
        num_channels: int,
        edge_dim: int,
        layer_kernels: DotDict,
        mixer_block: str = "edge_conv",
        mixer_kwargs: Optional[dict] = None,
        layer_norm: bool = True,
        low_rank_bias: bool = True,
        bias_rank: int = 128,
        bias_base_maps: int = 8,
        bias_lat_degree: int = 32,
        bias_lon_wavenumber: int = 32,
        **kwargs,
    ) -> None:
        del kwargs
        mixer_kwargs = dict(mixer_kwargs or {})
        super().__init__(
            num_channels=num_channels,
            layer_kernels=layer_kernels,
            mixer=lambda i, o: GraphSpatialMixer(i, o, edge_dim, layer_kernels, block=mixer_block, **mixer_kwargs),
            bias=lambda c: GraphLowRankBias(
                c,
                layer_kernels,
                rank=bias_rank,
                num_base_maps=bias_base_maps,
                lat_degree=bias_lat_degree,
                lon_wavenumber=bias_lon_wavenumber,
            ),
            layer_norm=layer_norm,
            low_rank_bias=low_rank_bias,
        )
