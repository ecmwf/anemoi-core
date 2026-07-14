# (C) Copyright 2025-2026 Anemoi contributors.
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


class NoiseLevelUncertainty(torch.nn.Module):
    """Learn one EDM2 log-variance value from each diffusion noise level.

    The head depends only on ``c_noise = log(sigma) / 4`` and is discarded at
    inference. Zero-initialising the projection makes enabling the head an exact
    no-op at initialisation.
    """

    def __init__(self, num_channels: int = 32, max_period: int = 10000) -> None:
        super().__init__()
        if num_channels <= 0 or num_channels % 2:
            msg = "num_channels must be a positive even integer."
            raise ValueError(msg)
        if max_period <= 0:
            raise ValueError("max_period must be positive.")

        self.embedding = SinusoidalEmbeddings(num_channels=num_channels, max_period=max_period)
        # nn.Linear initialisation consumes random numbers before the parameters
        # are reset. Restore the CPU RNG state so toggling EDM2 does not perturb
        # the denoiser's initialisation or the training data/noise stream.
        with torch.random.fork_rng(devices=[]):
            self.linear = torch.nn.Linear(num_channels, 1)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, c_noise: torch.Tensor) -> torch.Tensor:
        """Return flattened per-sample log-variance values in fp32.

        Parameters
        ----------
        c_noise : torch.Tensor
            EDM conditioning values of any shape, normally ``(batch, ensemble)``.

        Returns
        -------
        torch.Tensor
            One flattened log-variance value per input element.
        """
        with torch.autocast(device_type=c_noise.device.type, enabled=False):
            values = c_noise.reshape(-1, 1).float()
            return self.linear(self.embedding(values)).squeeze(-1)
