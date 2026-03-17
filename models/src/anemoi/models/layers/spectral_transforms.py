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
import torch.nn.functional as F

from anemoi.models.layers.spectral_helpers import SphericalHarmonicTransform

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
        patch_size: tuple[int, int] | None = None,
        patch_stride: tuple[int, int] | None = None,
        patch_padding: bool = False,
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
        patch_size: tuple[int, int] | None
            Optional patch size `(patch_y, patch_x)` for patch-wise FFT.
            If None, FFT is applied on the full `(y, x)` field.
        patch_stride: tuple[int, int] | None
            Optional patch stride `(stride_y, stride_x)` for patch-wise FFT.
            Defaults to `patch_size` (non-overlapping patches).
        patch_padding: bool
            If True, allow non-divisible `(y_dim, x_dim)` by zero-padding on the
            bottom/right edges before patch extraction.
        """
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.patch_size = patch_size
        self.patch_stride = patch_stride or patch_size
        self.patch_padding = patch_padding
        self.patch_pad_y = 0
        self.patch_pad_x = 0
        nodes_slice = nodes_slice or (0, None)  # we don't want einops to silently fail
        # by slicing random parts of the input
        self.nodes_slice = slice(*nodes_slice)
        self.apply_filter = apply_filter

        if self.patch_size is not None:
            patch_y, patch_x = self.patch_size
            if patch_y <= 0 or patch_x <= 0:
                raise ValueError("patch_size must contain strictly positive values")
            if patch_y > self.y_dim or patch_x > self.x_dim:
                raise ValueError(
                    f"patch_size {self.patch_size} must fit within full grid (y_dim={self.y_dim}, x_dim={self.x_dim})"
                )
            if self.patch_stride is None:
                self.patch_stride = self.patch_size
            stride_y, stride_x = self.patch_stride
            if stride_y <= 0 or stride_x <= 0:
                raise ValueError("patch_stride must contain strictly positive values")
            rem_y = (self.y_dim - patch_y) % stride_y
            rem_x = (self.x_dim - patch_x) % stride_x
            if rem_y != 0 or rem_x != 0:
                if not self.patch_padding:
                    raise ValueError(
                        "patch_size/patch_stride must tile the grid with integer patch counts "
                        "when patch_padding=False: "
                        f"got patch_size={self.patch_size}, patch_stride={self.patch_stride}, "
                        f"grid=({self.y_dim}, {self.x_dim})"
                    )
                self.patch_pad_y = (stride_y - rem_y) % stride_y
                self.patch_pad_x = (stride_x - rem_x) % stride_x

        if apply_filter:
            if self.patch_size is None:
                self.filter = self.lowpass_filter(x_dim, y_dim)
            else:
                patch_y, patch_x = self.patch_size
                self.filter = self.lowpass_filter(patch_x, patch_y)

    @staticmethod
    def lowpass_filter(x_dim: int, y_dim: int) -> torch.Tensor:
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
        data = torch.index_select(data, -2, torch.arange(*self.nodes_slice.indices(data.size(-2)), device=data.device))

        var = data.shape[-1]
        try:
            data = einops.rearrange(data, "... (y x) v -> ... y x v", x=self.x_dim, y=self.y_dim, v=var)
        except Exception as e:
            raise einops.EinopsError(
                f"Possible dimension mismatch in einops.rearrange in FFT2D layer: "
                f"expected (y * x) == last spatial dim with y={self.y_dim}, x={self.x_dim}"
            ) from e

        if self.patch_size is None:
            fft = torch.fft.fft2(data, dim=(-2, -3))
            if self.apply_filter:
                fft *= self.filter.to(device=data.device, dtype=data.dtype)
            return fft

        patch_y, patch_x = self.patch_size
        stride_y, stride_x = self.patch_stride
        lead_shape = data.shape[:-3]
        var = data.shape[-1]
        flat = data.reshape(-1, self.y_dim, self.x_dim, var)
        flat = einops.rearrange(flat, "n y x v -> n v y x")
        if self.patch_pad_y > 0 or self.patch_pad_x > 0:
            # Pad only bottom/right to preserve top-left alignment.
            flat = F.pad(flat, (0, self.patch_pad_x, 0, self.patch_pad_y))
        unfolded = F.unfold(flat, kernel_size=(patch_y, patch_x), stride=(stride_y, stride_x))
        n_patches = unfolded.shape[-1]
        patches = unfolded.transpose(1, 2).reshape(-1, n_patches, var, patch_y, patch_x)
        fft = torch.fft.fft2(patches, dim=(-2, -1))

        padded_y = self.y_dim + self.patch_pad_y
        padded_x = self.x_dim + self.patch_pad_x
        n_patches_y = (padded_y - patch_y) // stride_y + 1
        n_patches_x = (padded_x - patch_x) // stride_x + 1
        fft = fft.reshape(*lead_shape, n_patches_y, n_patches_x, var, patch_y, patch_x)
        fft = einops.rearrange(fft, "... py px v y x -> ... py px y x v")
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
        truncation: int | None = None,
        **kwargs,
    ) -> None:
        """SHT on a regular lon-lat grid.

        Parameters
        ----------
        nlat : int
            Number of latitudes in the regular grid.
        truncation : int | None
            Truncation parameter for the spherical harmonic transform. Keeping "truncation" wave numbers.
        """
        super().__init__()
        self.nlat = nlat
        self.nlon = 2 * self.nlat
        self.lons_per_lat = [self.nlon] * self.nlat
        self._sht = SphericalHarmonicTransform(
            lons_per_lat=self.lons_per_lat, truncation=truncation or self.nlat // 2 - 1
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        b, t, e, p, v = data.shape
        assert p == self._sht.n_grid_points, f"Input points={p} does not match expected nlat*nlon={self.nlat*self.nlon}"
        x = einops.rearrange(data, "b t e p v -> b t e v p")
        coeffs = self._sht(x)

        # -> [b,t,e,L,M,v] == [b,t,e,y_freq,x_freq,v]
        return einops.rearrange(coeffs, "(b t e v) yF xF -> b t e yF xF v", b=b, e=e, v=v, t=t)


class ReducedSHT(SpectralTransform):
    """SHT on a reduced Gaussian grid."""

    def __init__(
        self,
        grid: str,
        truncation: int | None = None,
        **kwargs,
    ) -> None:
        """SHT on a reduced Gaussian grid.

        Parameters
        ----------
        grid : str
            Name of the reduced Gaussian grid (e.g., "n320"). Only "n320" is currently supported.
        truncation : int | None
            Truncation parameter for the spherical harmonic transform. Keeping "truncation" wave numbers.
        """
        super().__init__()

        if grid not in ["n320", "N320"]:
            raise ValueError("Only the N320 reduced Gaussian grid SHT is supported.")
        else:
            self.nlat = 2 * int(grid[1:])  # N320 has 640 latitudes from pole to pole

        # Fetch regular grid data
        try:
            from anemoi.transform.grids.named import lookup
        except ImportError:
            raise ImportError(
                "anemoi.transform is required for ReducedSHT transform. Install optional dependencies: pip install anemoi-models[spectra]"
            )

        # To generate a grid
        # anemoi-transform get-grid --source mars grid=n320,levtype=sfc,param=2t grid-n320.npz

        lats = lookup(grid)["latitudes"]

        # Get latitudes of this grid
        unique_lats = sorted(set(lats))

        # Calculate longitudes per latitude
        self.lons_per_lat = [int((lats == unique_lat).sum()) for unique_lat in unique_lats]

        self._sht = SphericalHarmonicTransform(
            lons_per_lat=self.lons_per_lat, truncation=truncation or self.nlat // 2 - 1
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        b, t, e, p, v = data.shape
        assert p == self._sht.n_grid_points, f"Input points={p} does not match expected nlat*nlon={self.nlat*self.nlon}"
        x = einops.rearrange(data, "b t e p v -> b t e v p")
        coeffs = self._sht(x)

        # -> [b,t,e,L,M,v] == [b,t,e,y_freq,x_freq,v]
        return einops.rearrange(coeffs, "b t e v yF xF -> b t e yF xF v", b=b, e=e, v=v, t=t)


class OctahedralSHT(SpectralTransform):
    """SHT on an octahedral reduced grid."""

    def __init__(
        self,
        nlat: int,
        truncation: int | None = None,
        **kwargs,
    ) -> None:
        """SHT on an octahedral reduced grid.

        Parameters
        ----------
        nlat : int
            Number of latitudes in the octahedral grid. The number of longitudes per latitude will be determined based on the octahedral grid structure.
        truncation : int | None
            Truncation parameter for the spherical harmonic transform. Keeping "truncation" wave numbers.
        """
        super().__init__()
        self.nlat = nlat
        self.lons_per_lat = [20 + 4 * i for i in range(self.nlat // 2)]
        self.lons_per_lat += list(reversed(self.lons_per_lat))
        self._sht = SphericalHarmonicTransform(
            lons_per_lat=self.lons_per_lat, truncation=truncation or self.nlat // 2 - 1
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        b, t, e, p, v = data.shape
        assert (
            p == self._sht.n_grid_points
        ), f"Input points={p} does not match expected octahedral flattened rings={self._sht.n_grid_points}"

        # expects [..., points] where points is flattened spatial dim
        x = einops.rearrange(data, "b t e p v -> (b t e v) p")
        coeffs = self._sht(x)  # complex: (b*t*e*v, L, M)
        return einops.rearrange(coeffs, "(b t e v) yF xF -> b t e yF xF v", b=b, t=t, e=e, v=v)
