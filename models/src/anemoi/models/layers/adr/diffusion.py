# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Optional

from torch import Tensor
from torch import nn

from anemoi.models.layers.adr.layers import LowRankBias
from anemoi.models.layers.adr.layers import SpatialMixer
from anemoi.models.layers.adr.layers import init_branch
from anemoi.models.layers.layout import LatLonLayout
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


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
