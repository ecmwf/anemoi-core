# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math

import torch


class RandomFourierEmbeddings(torch.nn.Module):
    """Random fourier embeddings for noise levels."""

    def __init__(self, num_channels: int = 32, scale: int = 16):
        super().__init__()
        self.register_buffer("frequencies", torch.randn(num_channels // 2) * scale)
        self.register_buffer("pi", torch.tensor(math.pi))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.frequencies.unsqueeze(0) * 2 * self.pi
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class SinusoidalEmbeddings(torch.nn.Module):
    """Fourier embeddings for noise levels."""

    def __init__(self, num_channels: int = 32, max_period: int = 10000):
        super().__init__()
        zdim = num_channels // 2
        self.register_buffer("frequencies", torch.exp(-math.log(max_period) * torch.arange(0, zdim) / zdim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x[:] * self.frequencies
        return torch.cat((out.sin(), out.cos()), dim=-1)
