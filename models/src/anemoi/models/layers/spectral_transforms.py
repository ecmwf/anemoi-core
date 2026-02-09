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

from anemoi.models.layers.spectral_helpers import SphericalHarmonicTransform
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


class SpectralTransform(torch.nn.Module):
    """Abstract base class for spectral transforms."""

    @abc.abstractmethod
    def forward(
        self,
        data: torch.Tensor,
        **kwargs,
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
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        nodes_slice = nodes_slice or (0, None)  # we don't want einops to silently fail
        # by slicing random parts of the input
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
            data, TensorDim.GRID, torch.arange(*self.nodes_slice.indices(data.size(TensorDim.GRID)), device=data.device)
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
        b, t, e, points, v = data.shape
        assert points == self.x_dim * self.y_dim

        x = einops.rearrange(
            data,
            "b t e (y x) v -> (b t e v) y x",
            x=self.x_dim,
            y=self.y_dim,
        )
        x = dct_2d(x)
        return einops.rearrange(x, "(b t e v) y x -> b t e y x v", b=b, e=e, v=v, t=t)


class RegularSHT(SpectralTransform):
    """SHT on a regular lon-lat grid."""

    def __init__(
        self,
        nlat: int,
        nlon: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.nlat = nlat
        self.nlon = nlon
        self.lons_per_lat = [nlon] * nlat
        self._sht = SphericalHarmonicTransform(
            nlat=self.nlat, lons_per_lat=self.lons_per_lat, lmax=self.nlat // 2, mmax=self.nlat // 2
        )
        self.y_freq = self._sht.lmax
        self.x_freq = self._sht.mmax

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        b, t, e, p, v = data.shape
        assert p == self._sht.n_grid_points, f"Input points={p} does not match expected nlat*nlon={self.nlat*self.nlon}"
        x = einops.rearrange(data, "b t e (y x) v -> (b t e v) y x", y=self.nlat, x=self.nlon)
        coeffs = self._sht(x)

        # -> [b,t,e,L,M,v] == [b,t,e,y_freq,x_freq,v]
        return einops.rearrange(coeffs, "(b t e v) yF xF -> b t e yF xF v", b=b, e=e, v=v, t=t)


class ReducedSHT(SpectralTransform):
    """SHT on a reduced Gaussian grid."""

    def __init__(
        self,
        grid: str,
        **kwargs,
    ) -> None:
        super().__init__()

        # TODO implement logic for deriving nlat and lons_per_lat from given reduced grid spec
        # e.g. N320
        # self.nlat = nlat
        # self.lons_per_lat = [nlon] * nlat
        raise NotImplementedError("ReducedSHT is not yet implemented.")

        self._sht = SphericalHarmonicTransform(
            nlat=self.nlat, lons_per_lat=self.lons_per_lat, lmax=self.nlat // 2, mmax=self.nlat // 2
        )
        self.y_freq = self._sht.lmax
        self.x_freq = self._sht.mmax

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        b, t, e, p, v = data.shape
        assert p == self._sht.n_grid_points, f"Input points={p} does not match expected nlat*nlon={self.nlat*self.nlon}"
        x = einops.rearrange(data, "b t e (y x) v -> (b t e v) y x", y=self.nlat, x=self.nlon)
        coeffs = self._sht(x)

        # -> [b,t,e,L,M,v] == [b,t,e,y_freq,x_freq,v]
        return einops.rearrange(coeffs, "(b t e v) yF xF -> b t e yF xF v", b=b, e=e, v=v, t=t)


class OctahedralSHT(SpectralTransform):
    """SHT on an octahedral reduced grid."""

    def __init__(
        self,
        nlat: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.nlat = nlat
        self.lons_per_lat = [20 + 4 * i for i in range(self.nlat // 2)]
        self.lons_per_lat += list(reversed(self.lons_per_lat))
        self._sht = SphericalHarmonicTransform(
            nlat=self.nlat, lons_per_lat=self.lons_per_lat, lmax=self.nlat // 2, mmax=self.nlat // 2
        )
        self.y_freq = self._sht.lmax
        self.x_freq = self._sht.mmax

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        b, t, e, p, v = data.shape
        assert (
            p == self._sht.n_grid_points
        ), f"Input points={p} does not match expected octahedral flattened rings={self._sht.n_grid_points}"

        # expects [..., points] where points is flattened spatial dim
        x = einops.rearrange(data, "b t e p v -> (b t e v) p")
        coeffs = self._sht(x)  # complex: (b*t*e*v, L, M)
        return einops.rearrange(coeffs, "(b t e v) yF xF -> b t e yF xF v", b=b, t=t, e=e, v=v)
