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
import numpy as np
import torch
import torch.fft

from anemoi.models.layers.spectral_helpers import CartesianRealSHT
from anemoi.models.layers.spectral_helpers import EcTransOctahedralSHTModule
from anemoi.models.layers.spectral_helpers import OctahedralRealSHT
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


class SpectralTransform(torch.nn.Module):
    """Base class for spectral transforms."""

    @abc.abstractmethod
    def forward(
        self,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """Transform data to spectral domain.

        Parameters
        ----------
        data : torch.Tensor
            Input data in the spatial domain of expected shape
            `[batch, ensemble, points, variables]`.

        Returns
        -------
        torch.Tensor
            Data transformed to the spectral domain, of shape
            `[batch, ensemble, y_freq, x_freq, variables]`.
        """


class FFT2D(SpectralTransform):
    """2D Fast Fourier Transform (FFT) implementation."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        apply_filter: bool = True,
        nodes_slice: tuple[int, int | None] | None = None,
        **kwargs,
    ) -> None:
        """2D FFT Transform.

        Parameters
        ----------
        x_dim : int
            size of the spatial dimension x of the original data in 2D
        y_dim : int
            size of the spatial dimension y of the original data in 2D
        apply_filter: bool
            Apply low-pass filter to ignore frequencies beyond the Nyquist limit
        """
        self.x_dim = x_dim
        self.y_dim = y_dim
        nodes_slice = nodes_slice or (0, None)
        self.nodes_slice = slice(*nodes_slice)
        self.apply_filter = apply_filter
        if apply_filter:
            self.filter = self.lowpass_filter(x_dim, y_dim)

    @staticmethod
    def lowpass_filter(x_dim, y_dim):
        fx = torch.fft.fftfreq(x_dim)
        fy = torch.fft.fftfreq(y_dim)

        KX, KY = torch.meshgrid(fx, fy, indexing="ij")
        k = torch.sqrt(KX * KX + KY * KY)

        mask = k < 0.5  # torch.where(k < 0.5, 1.0 - 2.0 * k, 0.0)
        return einops.rearrange(mask, "x y -> y x 1")

    def forward(
        self,
        data: torch.Tensor,
    ) -> torch.Tensor:
        data = torch.index_select(
            data, TensorDim.GRID, torch.arange(*self.nodes_slice.indices(data.size(TensorDim.GRID)))
        )
        var = data.shape[-1]
        try:
            data = einops.rearrange(data, "... (y x) v -> ... y x v", x=self.x_dim, y=self.y_dim, v=var)
        except Exception as e:
            raise einops.EinopsError(
                f"Possible dimension mismatch in einops.rearrange in FFT2D layer: "
                f"expected (y * x) == last spatial dim with y={self.y_dim}, x={self.x_dim}"
            ) from e

        fft = torch.fft.fft2(data, dim=(-2, -3))
        if self.apply_filter:
            fft *= self.filter.to(device=data.device, dtype=data.dtype)
        return fft


class DCT2D(SpectralTransform):
    """2D Discrete Cosine Transform."""

    def __init__(self, x_dim: int, y_dim: int, **kwargs) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        try:
            from torch_dct import dct_2d
        except ImportError:
            raise ImportError("torch_dct is required for DCT2D transform. ")
        b, e, points, v = data.shape
        assert points == self.x_dim * self.y_dim

        x = einops.rearrange(
            data,
            "b e (y x) v -> (b e v) y x",
            x=self.x_dim,
            y=self.y_dim,
        )
        x = dct_2d(x)
        return einops.rearrange(
            x,
            "(b e v) y x -> b e y x v",
            b=b,
            e=e,
            v=v,
        )


class CartesianSHT(SpectralTransform):
    """SHT on a regular (y_dim=nlat, x_dim=nlon) grid."""

    def __init__(
        self,
        x_dim: int,  # nlon
        y_dim: int,  # nlat
        grid: str = "legendre-gauss",
        **kwargs,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.grid = grid
        self._sht = CartesianRealSHT(nlat=self.y_dim, nlon=self.x_dim, grid=self.grid)
        self.y_freq = self._sht.lmax
        self.x_freq = self._sht.mmax

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        b, e, p, v = data.shape
        assert (
            p == self.x_dim * self.y_dim
        ), f"Input points={p} does not match expected y_dim*x_dim={self.y_dim*self.x_dim}"
        x = einops.rearrange(data, "b e (y x) v -> (b e v) y x", y=self.y_dim, x=self.x_dim)
        coeffs = self._sht(x)

        # -> [b,e,L,M,v] == [b,e,y_freq,x_freq,v]
        return einops.rearrange(coeffs, "(b e v) yF xF -> b e yF xF v", b=b, e=e, v=v)


class OctahedralSHT(SpectralTransform):
    """SHT on an octahedral reduced grid."""

    def __init__(
        self,
        nlat: int,
        lmax: int | None = None,
        mmax: int | None = None,
        folding: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.nlat = nlat
        self.lmax = lmax
        self.mmax = mmax
        self.folding = folding
        self._sht = OctahedralRealSHT(nlat=self.nlat, lmax=self.lmax, mmax=self.mmax, folding=folding)
        self._nlon = self._sht.nlon
        self._expected_points = int(np.sum(self._nlon))
        self.y_freq = self._sht.lmax
        self.x_freq = self._sht.mmax

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        b, e, p, v = data.shape
        assert (
            p == self._expected_points
        ), f"Input points={p} does not match expected octahedral flattened rings={self._expected_points}"

        # expects [..., points] where points is flattened spatial dim
        x = einops.rearrange(data, "b e p v -> (b e v) p")
        coeffs = self._sht(x)  # complex: (b*e*v, L, M)
        tmp = einops.rearrange(coeffs, "(b e v) yF xF -> b e yF xF v", b=b, e=e, v=v)
        return tmp


class EcTransOctahedralSHT(SpectralTransform):
    def __init__(
        self,
        truncation: int,
        dtype: torch.dtype = torch.float32,
        filepath: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.truncation = int(truncation)
        self.dtype = dtype

        self._sht = EcTransOctahedralSHTModule(truncation=self.truncation, dtype=self.dtype, filepath=filepath)
        self._expected_points = int(self._sht.n_grid_points)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        _, _, points, _ = data.shape
        assert points == self._expected_points, (
            f"Input data spatial dimension {points} does not match expected "
            f"size {self._expected_points} for truncation={self.truncation}."
        )
        return self._sht(data)
