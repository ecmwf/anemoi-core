# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from anemoi.models.layers.layout import LatLonLayout


def geocyclic_pad(x: Tensor, pad: int) -> Tensor:
    """Pad a field of shape (..., nlat, nlon) so that it carries on across the poles and around in longitude.

    A path that goes over a pole comes down the other side half a turn further round, so the rows
    added beyond a pole are the rows next to it, in reverse order and shifted by half the longitudes.
    Columns wrap around. Needs an even ``nlon``.
    """
    half_turn = x.shape[-1] // 2
    top = x[..., :pad, :].flip(-2).roll(half_turn, dims=-1)
    bottom = x[..., -pad:, :].flip(-2).roll(half_turn, dims=-1)
    x = torch.cat([top, x, bottom], dim=-2)
    return torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)


class GeocyclicDepthwiseConv(nn.Module):
    """A learned square stencil applied to every channel on its own, continuing across the poles."""

    def __init__(self, num_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.num_channels = num_channels
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size, groups=num_channels, padding=0, bias=False)

    def forward(self, x: Tensor, layout: "LatLonLayout") -> Tensor:
        # Each GPU applies the stencil to the whole globe for its share of the channels, using
        # its share of the stencil weights.
        start, end = layout.local_groups(self.num_channels)
        y = layout.to_globe(x.view(-1, self.num_channels, 1))
        y = geocyclic_pad(y.flatten(1, 2), self.pad)
        y = F.conv2d(y, self.conv.weight[start:end], groups=end - start)
        y = y.view(layout.batch_size, end - start, 1, layout.nlat, layout.nlon)
        return layout.from_globe(y, self.num_channels).view(-1, self.num_channels)
