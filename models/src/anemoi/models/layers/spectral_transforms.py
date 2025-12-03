# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import abc
import logging

import einops
import torch
import torch.fft

LOGGER = logging.getLogger(__name__)


class SpectralTransform:
    """Abstract base class for spectral transforms."""

    @abc.abstractmethod
    def __call__(
        self,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """Transform data to spectral domain.

        Parameters
        ----------
        data : torch.Tensor
            Input data in the spatial domain of expected shape
            `[batch, time, ensemble, points, variables]`.

        Returns
        -------
        torch.Tensor
            Data transformed to the spectral domain, of shape
            `[batch, time, ensemble, y_freq, x_freq, variables]`.
        """


class FFT2D(SpectralTransform):
    """2D Fast Fourier Transform (FFT) implementation."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        nodes_slice: slice = slice(None),  # TODO: generic indexing class
    ) -> None:
        """2D FFT Transform.

        Parameters
        ----------
        x_dim : int
            size of the spatial dimension x of the original data in 2D
        y_dim : int
            size of the spatial dimension y of the original data in 2D
        """
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.nodes_slice = nodes_slice

    def __call__(
        self,
        data: torch.Tensor,
    ) -> torch.Tensor:
        data = data[:, :, :, self.nodes_slice, :]
        batch_size, time, ens, _, var = data.shape
        # [batch, time, ens, y*x, variables] -> [batch*time*ens*variables, y, x]
        data = einops.rearrange(data, "b t e (y x) v -> (b t e v) y x", x=self.x_dim, y=self.y_dim)
        fft_data = torch.fft.fft2(data)
        # [batch*time*ens*variables, y, x] -> [batch, time, ens y, x, variables]
        return einops.rearrange(fft_data, "(b t e v) y x -> b t e y x v", b=batch_size, t=time, e=ens, v=var)


class SHT(SpectralTransform):
    """Placeholder for Spherical Harmonics Transform."""

    def __call__(
        self,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """Transform data to spectral domain using spherical harmonics.

        Parameters
        ----------
        data : torch.Tensor
            Input data in the spatial domain.

        Returns
        -------
        torch.Tensor
            Data transformed to the spectral domain.
        """
        msg = "Spherical harmonics transform is not implemented yet."
        raise NotImplementedError(msg)
