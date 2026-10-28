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

from anemoi.models.layers.adr.layers import init_branch
from anemoi.models.layers.layout import Layout
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class Diffusion(nn.Module):
    """Diffusion step: layer norm, a spatial mixer and a learned bias field.

    The mixer and the bias come from the processor's geometry, so the same step runs on a regular
    latitude-longitude grid and on a mesh.
    """

    def __init__(
        self,
        *,
        num_channels: int,
        layer_kernels: DotDict,
        geometry,
        layer_norm: bool = True,
        low_rank_bias: bool = True,
        **kwargs,
    ) -> None:
        del kwargs
        super().__init__()
        self.layer_norm = layer_kernels.LayerNorm(normalized_shape=num_channels) if layer_norm else None
        self.mixer = geometry.spatial_mixer(num_channels, num_channels, layer_kernels)
        self.bias = geometry.position_bias(num_channels, layer_kernels) if low_rank_bias else None
        init_branch(*self.mixer.branch_layers())

    def forward(self, x: Tensor, layout: Layout, cond: Optional[Tensor] = None) -> Tensor:
        """Forward pass of the diffusion step.

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
        cond_kwargs = {"cond": cond} if cond is not None else {}
        if self.layer_norm is not None:
            x = self.layer_norm(x, **cond_kwargs)
        x = self.mixer(x, layout, cond)
        if self.bias is not None:
            x = self.bias(x, layout)
        return x
