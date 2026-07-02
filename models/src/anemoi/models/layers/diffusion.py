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


class NoiseLevelUncertainty(torch.nn.Module):
    """EDM2 uncertainty head u(sigma): a 1-D function of the noise level only.

    Implements the per-noise-level uncertainty estimate from EDM2
    (Karras et al., "Analyzing and Improving the Training Dynamics of Diffusion
    Models", 2024). The noise level ``c_noise`` (== ln(sigma) / 4) is embedded with
    the same sinusoidal embedding used to condition the denoiser, then mapped through
    a single zero-initialised Linear to ONE scalar per sample: ``logvar = u_raw``.

    The denoiser loss is reformulated as a Gaussian negative-log-likelihood
    ``L_eff = L_raw / exp(logvar) + logvar``; at the optimum ``exp(logvar) -> E[L_raw(sigma)]``
    so the denoiser gradient is rescaled by ~1/E[L_raw(sigma)] per noise level, which
    auto-balances the noise levels without hand-tuning P_mean / P_std.

    This head is NOT per-pixel and NOT per-variable: it depends only on sigma.
    At inference it is discarded.

    Design notes
    ------------
    * Zero-init of BOTH the Linear weight and bias makes ``logvar`` identically 0 at
      step 0, so ``exp(logvar) == 1`` and ``L_eff == L_raw`` -- a true no-op at init.
    * The Linear parameters are created inside ``torch.random.fork_rng`` so that
      *enabling* the head draws zero net global RNG. This keeps the denoiser's
      noise/data realisation bit-identical to the ``enabled=False`` baseline, which is
      what makes the no-op-at-init numerically exact (not merely approximate).
    * The forward runs in fp32 (autocast disabled) regardless of the surrounding
      bf16 autocast region, matching the EDM2 reference.
    """

    def __init__(self, num_channels: int = 32, max_period: int = 10000):
        super().__init__()
        self.embedding = SinusoidalEmbeddings(num_channels=num_channels, max_period=max_period)
        # Build the head without perturbing the global RNG stream (nn.Linear's default
        # reset_parameters draws from it); fork_rng saves/restores CPU RNG so the draw
        # is rolled back. The params are then zero-initialised.
        with torch.random.fork_rng(devices=[]):
            self.linear = torch.nn.Linear(num_channels, 1)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, c_noise: torch.Tensor) -> torch.Tensor:
        """Map noise levels to per-sample log-variance.

        Parameters
        ----------
        c_noise : torch.Tensor
            Conditioning ``ln(sigma) / 4`` of any shape; flattened to one value per
            sample (e.g. ``(batch, ensemble)`` -> ``(batch * ensemble,)``).

        Returns
        -------
        torch.Tensor
            ``logvar`` of shape ``(batch * ensemble,)`` (fp32).
        """
        with torch.autocast(device_type=c_noise.device.type, enabled=False):
            x = c_noise.reshape(-1, 1).float()
            return self.linear(self.embedding(x)).squeeze(-1)
