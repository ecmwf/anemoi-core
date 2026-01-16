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

LOGGER = logging.getLogger(__name__)


class SpectralTransform(torch.nn.Module):
    """Abstract base class for spectral transforms."""

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
        nodes_slice: tuple[int, int | None] | None = None,
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
        nodes_slice = nodes_slice or (0, None)
        self.nodes_slice = slice(*nodes_slice)

    def __call__(
        self,
        data: torch.Tensor,
    ) -> torch.Tensor:
        data = data[:, :, self.nodes_slice, :]
        batch_size, ens, _, var = data.shape
        assert data.shape[2] == self.x_dim * self.y_dim, (
            f"Input data spatial dimension {data.shape[2]} does not match expected "
            f"size {self.x_dim * self.y_dim} from x_dim and y_dim"
        )
        # [batch, ens, y*x, variables] -> [batch*ens*variables, y, x]
        # TODO (Ophelia): edit this when multi ouptuts get merged
        data = einops.rearrange(data, "b e (y x) v -> (b e v) y x", x=self.x_dim, y=self.y_dim)
        fft_data = torch.fft.fft2(data)
        # [batch**ens*variables, y, x] -> [batch, ens y, x, variables]
        return einops.rearrange(fft_data, "(b e v) y x -> b e y x v", b=batch_size, e=ens, v=var)


class DCT2D(SpectralTransform):
    """2D Discrete Cosine Transform."""

    def __init__(self, x_dim: int, y_dim: int) -> None:
        self.x_dim = x_dim
        self.y_dim = y_dim

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
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


class SHT(SpectralTransform):
    """Placeholder for Spherical Harmonics Transform."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

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
        msg = "Each spherical harmonics transform subclass " "should implement the __call__ method."
        raise NotImplementedError(msg)


class CartesianSHT(SHT):
    """SHT on a regular (y_dim=nlat, x_dim=nlon) grid."""

    def __init__(
        self,
        x_dim: int,  # nlon
        y_dim: int,  # nlat
        grid: str = "legendre-gauss",
        nodes_slice: tuple[int, int | None] | None = None,
    ) -> None:
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.grid = grid
        nodes_slice = nodes_slice or (0, None)
        self.nodes_slice = slice(*nodes_slice)
        self._sht = CartesianRealSHT(nlat=self.y_dim, nlon=self.x_dim, grid=self.grid)
        self.y_freq = self._sht.lmax
        self.x_freq = self._sht.mmax

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        # TODO (Ophelia): change this when multi outputs get merged
        data = data[:, :, self.nodes_slice, :]

        if self.nodes_slice != slice(0, None):
            raise NotImplementedError(
                "nodes_slice is not supported for CartesianSHT unless it is the full grid. "
                "SHT requires a complete (nlat*nlon) grid ordering."
            )

        b, e, p, v = data.shape
        assert (
            p == self.x_dim * self.y_dim
        ), f"Input points={p} does not match expected y_dim*x_dim={self.y_dim*self.x_dim}"
        x = einops.rearrange(data, "b e (y x) v -> (b e v) y x", y=self.y_dim, x=self.x_dim)
        coeffs = self._sht(x)

        # -> [b,e,L,M,v] == [b,e,y_freq,x_freq,v]
        return einops.rearrange(coeffs, "(b e v) yF xF -> b e yF xF v", b=b, e=e, v=v)


class OctahedralSHT(SHT):
    """SHT on an octahedral reduced grid."""

    def __init__(
        self,
        y_dim: int,
        x_dim: int | None = None,  # ignored, kept for signature-uniformity
        lmax: int | None = None,
        mmax: int | None = None,
        folding: bool = False,
        nodes_slice: tuple[int, int | None] | None = None,
    ) -> None:
        self.y_dim = y_dim
        self.x_dim = x_dim  # unused but present to match FFT2D/CartesianSHT signature
        self.lmax = lmax
        self.mmax = mmax
        self.folding = folding
        nodes_slice = nodes_slice or (0, None)
        self.nodes_slice = slice(*nodes_slice)
        self._sht = OctahedralRealSHT(nlat=self.y_dim, lmax=lmax, mmax=mmax, folding=folding)
        self._nlon = self._sht.nlon
        self._expected_points = int(np.sum(self._nlon))
        self.y_freq = self._sht.lmax
        self.x_freq = self._sht.mmax

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        # TODO (Ophelia): change this when multi outputs get merged
        data = data[:, :, self.nodes_slice, :]

        if self.nodes_slice != slice(0, None):
            raise NotImplementedError(
                "nodes_slice is not supported for OctahedralSHT unless it is the full grid. "
                "Octahedral SHT expects full ring structure."
            )

        b, e, p, v = data.shape
        assert (
            p == self._expected_points
        ), f"Input points={p} does not match expected octahedral flattened rings={self._expected_points}"

        # expects [..., points] where points is flattened spatial dim
        x = einops.rearrange(data, "b e p v -> (b e v) p")
        coeffs = self._sht(x)  # complex: (b*e*v, L, M)
        return einops.rearrange(coeffs, "(b e v) yF xF -> b e yF xF v", b=b, e=e, v=v)


class EcTransOctahedralSHT(SHT):
    def __init__(
        self,
        truncation: int,
        dtype: torch.dtype = torch.float32,
        filepath: str | None = None,
        nodes_slice: tuple[int, int | None] | None = None,
        *,
        # optional “compat” args:
        x_dim: int | None = None,  # interpreted as max_nlon
        y_dim: int | None = None,  # interpreted as nlat (full globe)
    ) -> None:
        self.truncation = int(truncation)
        self.dtype = dtype
        self.y_dim = 2 * (self.truncation + 1)  # nlat full globe
        self.x_dim = 20 + 4 * self.truncation  # max nlon on any latitude ring

        if y_dim is not None and int(y_dim) != self.y_dim:
            raise ValueError(f"y_dim={y_dim} incompatible with truncation={self.truncation} (expected {self.y_dim}).")
        if x_dim is not None and int(x_dim) != self.x_dim:
            raise ValueError(f"x_dim={x_dim} incompatible with truncation={self.truncation} (expected {self.x_dim}).")

        nodes_slice = nodes_slice or (0, None)
        self.nodes_slice = slice(*nodes_slice)
        if not (self.nodes_slice.start == 0 and self.nodes_slice.stop is None):
            raise ValueError("EcTransOctahedralSHT does not support `nodes_slice` (requires full grid).")

        self._sht = EcTransOctahedralSHTModule(truncation=self.truncation, dtype=self.dtype, filepath=filepath)
        self._expected_points = int(self._sht.n_grid_points)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        data = data[:, :, self.nodes_slice, :]
        _, _, points, _ = data.shape
        assert points == self._expected_points, (
            f"Input data spatial dimension {points} does not match expected "
            f"size {self._expected_points} for truncation={self.truncation}."
        )
        return self._sht(data)
