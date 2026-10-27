# (C) Copyright 2026 Anemoi contributors.
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
from anemoi.models.layers.adr.layers import init_branch
from anemoi.models.layers.layout import LatLonLayout
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


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
