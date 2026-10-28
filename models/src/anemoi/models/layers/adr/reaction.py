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

from anemoi.models.layers.adr.layers import init_branch
from anemoi.models.layers.layout import Layout
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class Reaction(nn.Module):
    """Reaction step: layer norm and a pointwise MLP, with a learned bias field after its first layer.

    Only the bias depends on the geometry, which the processor passes in.
    """

    def __init__(
        self,
        *,
        num_channels: int,
        layer_kernels: DotDict,
        geometry,
        hidden_dim: int = 896,
        num_layers: int = 4,
        layer_norm: bool = True,
        low_rank_bias: bool = True,
        **kwargs,
    ) -> None:
        del kwargs
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be at least 2, got {num_layers}.")
        self.layer_norm = layer_kernels.LayerNorm(normalized_shape=num_channels) if layer_norm else None
        dims = [num_channels] + [hidden_dim] * (num_layers - 1) + [num_channels]
        self.layers = nn.ModuleList(layer_kernels.Linear(dims[i], dims[i + 1]) for i in range(num_layers))
        self.bias = geometry.position_bias(hidden_dim, layer_kernels) if low_rank_bias else None
        self.activation = layer_kernels.Activation()
        init_branch(*self.layers)

    def forward(self, x: Tensor, layout: Layout, cond: Optional[Tensor] = None) -> Tensor:
        """Forward pass of the reaction step.

        Parameters
        ----------
        x : Tensor
            The state, laid out as the processor's state: (batch, local rows, nlon, num_channels)
            on a latitude-longitude grid, or (batch * local nodes, num_channels) on a mesh.
        layout : Layout
            The layout of this forward pass, from the processor's geometry.
        cond : Tensor, optional
            Conditioning tensor, shaped like x with the condition channels, by default None.

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
