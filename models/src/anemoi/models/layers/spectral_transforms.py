# (C) Copyright 2025-2026 Anemoi contributors.
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

from anemoi.models.layers.spectral_helpers import InverseSphericalHarmonicTransform
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
        kwargs : dict
            Additional keyword arguments for the transform.

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
        apply_filter: bool = False,
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

    def prepare_for_fft(self, data: torch.Tensor) -> torch.Tensor:
        """Reshape data from flat ``(nodes, vars)`` to ``(y, x, vars)``."""
        var = data.shape[-1]
        try:
            return einops.rearrange(data, "... (y x) v -> ... y x v", x=self.x_dim, y=self.y_dim, v=var)
        except Exception as e:
            raise einops.EinopsError(
                f"Possible dimension mismatch in einops.rearrange in FFT2D layer: "
                f"expected (y * x) == last spatial dim with y={self.y_dim}, x={self.x_dim}"
            ) from e

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

        data = self.prepare_for_fft(data)

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


class SHT(SpectralTransform):
    """Spherical Harmonic Transform (SHT) baseclass."""

    def power_spectral_density(self, spectral_coeffs: torch.Tensor) -> torch.Tensor:
        """Return per-L power spectral density: sum over M of |coeff|^2."""
        return (spectral_coeffs.real**2 + spectral_coeffs.imag**2).sum(dim=-2)

    def cross_spectral_density(self, spectral_coeffs_a: torch.Tensor, spectral_coeffs_b: torch.Tensor) -> torch.Tensor:
        """Return per-L cross-spectral density."""
        return (spectral_coeffs_a.real * spectral_coeffs_b.real + spectral_coeffs_a.imag * spectral_coeffs_b.imag).sum(
            dim=-2
        )


class RegularSHT(SHT):
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
        return einops.rearrange(coeffs, "b t e v yF xF -> b t e yF xF v", b=b, e=e, v=v, t=t)


class ReducedSHT(SHT):
    """SHT on a reduced Gaussian grid."""

    def __init__(
        self,
        grid: str,
        truncation: int | None = None,
        use_graphed_rfft: bool = False,
        **kwargs,
    ) -> None:
        """SHT on a reduced Gaussian grid.

        Parameters
        ----------
        grid : str
            Name of the reduced Gaussian grid (e.g., "n320"). Only "n320" is currently supported.
        truncation : int | None
            Truncation parameter for the spherical harmonic transform. Keeping "truncation" wave numbers.
        use_graphed_rfft : bool
            Whether to use a graphed implementation of the rfft on reduced grids, which can be faster but may have
            higher memory usage and may not be supported by all devices.
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
            lons_per_lat=self.lons_per_lat,
            truncation=truncation or self.nlat // 2 - 1,
            use_graphed_rfft=use_graphed_rfft,
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        b, t, e, p, v = data.shape
        assert p == self._sht.n_grid_points, f"Input points={p} does not match expected nlat*nlon={self.nlat*self.nlon}"
        x = einops.rearrange(data, "b t e p v -> b t e v p")
        coeffs = self._sht(x)

        # -> [b,t,e,L,M,v] == [b,t,e,y_freq,x_freq,v]
        return einops.rearrange(coeffs, "b t e v yF xF -> b t e yF xF v", b=b, e=e, v=v, t=t)


class OctahedralSHT(SHT):
    """SHT on an octahedral reduced grid."""

    def __init__(
        self,
        nlat: int,
        truncation: int | None = None,
        use_graphed_rfft: bool = False,
        **kwargs,
    ) -> None:
        """SHT on an octahedral reduced grid.

        Parameters
        ----------
        nlat : int
            Number of latitudes in the octahedral grid. The number of longitudes per latitude will be determined based
            on the octahedral grid structure.
        truncation : int | None
            Truncation parameter for the spherical harmonic transform. Keeping "truncation" wave numbers.
        use_graphed_rfft : bool
            Whether to use a graphed implementation of the rfft on reduced grids, which can be faster but may have higher memory usage and may not be supported by all devices.
        """
        super().__init__()
        self.nlat = nlat
        self.lons_per_lat = [20 + 4 * i for i in range(self.nlat // 2)]
        self.lons_per_lat += list(reversed(self.lons_per_lat))
        self._sht = SphericalHarmonicTransform(
            lons_per_lat=self.lons_per_lat,
            truncation=truncation or self.nlat // 2 - 1,
            use_graphed_rfft=use_graphed_rfft,
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


class InverseSpectralTransform(torch.nn.Module):
    """Abstract base class for inverse spectral transforms.

    Input: complex tensor [..., l, m] (spectral coefficients).
    Output: real tensor [..., points] (spatial domain).
    """

    @abc.abstractmethod
    def forward(self, data: torch.Tensor) -> torch.Tensor: ...


class InverseRegularSHT(InverseSpectralTransform):
    """Inverse SHT on a regular lon-lat grid."""

    def __init__(self, nlat: int, truncation: int | None = None, **kwargs) -> None:
        """Initialize InverseRegularSHT.

        Parameters
        ----------
        nlat : int
            Number of latitudes.
        truncation : int | None
            Spectral truncation. Defaults to ``nlat // 2 - 1``.
        **kwargs : dict
            Additional keyword arguments (ignored).
        """
        super().__init__()
        self.nlat = nlat
        self.nlon = 2 * nlat
        self.lons_per_lat = [self.nlon] * self.nlat
        self._isht = InverseSphericalHarmonicTransform(
            lons_per_lat=self.lons_per_lat, truncation=truncation or self.nlat // 2 - 1
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self._isht(data)


class InverseReducedSHT(InverseSpectralTransform):
    """Inverse SHT on a reduced Gaussian grid."""

    def __init__(
        self,
        grid: str,
        truncation: int | None = None,
        use_graphed_irfft: bool = False,
        **kwargs,
    ) -> None:
        """Inverse SHT on a reduced Gaussian grid.

        Parameters
        ----------
        grid : str
            Name of the reduced Gaussian grid (e.g., "n320"). Only "n320" is currently supported.
        truncation : int | None
            Truncation parameter for the spherical harmonic transform. Keeping "truncation" wave numbers.
        use_graphed_irfft : bool
            Whether to use a graphed implementation of the irfft on reduced grids, which can be faster but may have
            higher memory usage and may not be supported by all devices.
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
                "anemoi.transform is required for InverseReducedSHT transform. Install optional dependencies: pip install anemoi-models[spectra]"
            )

        # To generate a grid
        # anemoi-transform get-grid --source mars grid=n320,levtype=sfc,param=2t grid-n320.npz

        lats = lookup(grid)["latitudes"]

        # Get latitudes of this grid
        unique_lats = sorted(set(lats))

        # Calculate longitudes per latitude
        self.lons_per_lat = [int((lats == unique_lat).sum()) for unique_lat in unique_lats]

        self._isht = InverseSphericalHarmonicTransform(
            lons_per_lat=self.lons_per_lat,
            truncation=truncation or self.nlat // 2 - 1,
            use_graphed_irfft=use_graphed_irfft,
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self._isht(data)


class InverseOctahedralSHT(InverseSpectralTransform):
    """Inverse SHT on an octahedral reduced grid."""

    def __init__(
        self,
        nlat: int,
        truncation: int | None = None,
        use_graphed_irfft: bool = False,
        **kwargs,
    ) -> None:
        """Inverse SHT on an octahedral reduced grid.

        Parameters
        ----------
        nlat : int
            Number of latitudes.
        truncation : int | None
            Spectral truncation. Defaults to nlat // 2 - 1.
        use_graphed_irfft : bool
            Whether to use a graphed implementation of the irfft on reduced grids, which can be faster but may have
            higher memory usage and may not be supported by all devices.
        **kwargs : dict
            Additional keyword arguments (ignored).
        """
        super().__init__()
        self.nlat = nlat
        self.lons_per_lat = [20 + 4 * i for i in range(self.nlat // 2)]
        self.lons_per_lat += list(reversed(self.lons_per_lat))
        self._isht = InverseSphericalHarmonicTransform(
            lons_per_lat=self.lons_per_lat,
            truncation=truncation or self.nlat // 2 - 1,
            use_graphed_irfft=use_graphed_irfft,
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self._isht(data)


class SphericalSpectralFilter(torch.nn.Module):
    r"""Isotropic spectral filter on a Gaussian grid.

    Analyses a field into spherical harmonics, scales every coefficient of degree
    :math:`l` by a response :math:`r_l`, and synthesises it back. Because the
    response depends on :math:`l` only, the filter is isotropic: it treats every
    location and direction on the sphere alike, which a filter applied on the grid
    itself would not.

    Its main use is restoring the spectral content of a product of fields.
    Multiplying two fields in grid space convolves their spectra, so the product
    carries energy at degrees neither factor had, and only a filter in spectral
    space can put it back in a prescribed band.

    The filter is built at its own ``truncation`` rather than the full resolution
    of the grid: the Legendre tables scale as ``truncation**2 * nlat``, so a filter
    that only needs low degrees should not pay for the high ones. Coefficients it
    produces are directly usable at any larger truncation by zero padding, since
    the basis does not depend on the truncation.

    Parameters
    ----------
    lons_per_lat : list[int]
        Number of longitudinal points on each latitude ring, from pole to pole.
    truncation : int
        Highest degree retained, so coefficients have ``truncation + 1`` degrees.
    """

    def __init__(self, lons_per_lat: list[int], truncation: int) -> None:
        super().__init__()

        lons_per_lat = [int(n) for n in lons_per_lat]
        nlat = len(lons_per_lat)
        self.truncation = truncation
        self.n_grid_points = sum(lons_per_lat)

        self._sht = SphericalHarmonicTransform(lons_per_lat=lons_per_lat, truncation=truncation)
        self._isht = InverseSphericalHarmonicTransform(lons_per_lat=lons_per_lat, truncation=truncation)
        # legpoly builds these in float64; the per-call `.to(x.dtype)` in the transforms
        # would otherwise copy them on every call.
        self._sht.weight = self._sht.weight.to(torch.float32)
        self._isht.pct = self._isht.pct.to(torch.float32)

        # Parseval: the area mean of a synthesised field's square is a weighted sum of its
        # coefficients' power, counting m > 0 twice (they stand for +m and -m). The scale of
        # the basis is measured, as for the round trip, rather than assumed.
        power_weight = torch.full((self.lmax, self.lmax), 2.0).tril()
        power_weight[:, 0] = 1.0
        unit = torch.zeros(1, self.lmax, self.lmax, dtype=torch.complex64)
        unit[0, 0, 0] = 1.0
        with torch.no_grad():
            constant = self.synthesise(unit)[0, 0].item()
        self.register_buffer("power_weight", power_weight * constant**2, persistent=False)

        # How high a degree a grid analyses exactly depends on its ring layout: a linear
        # reduced Gaussian grid such as N320 is exact up to nlat - 1, an octahedral one only
        # to about nlat / 2. Measure it rather than infer it from the grid type.
        self.round_trip_error = self._round_trip_error()
        if self.round_trip_error > 1e-4:
            LOGGER.warning(
                "SphericalSpectralFilter: this %d-ring grid does not analyse degrees up to %d exactly "
                "(round-trip error %.1e); the filtered field will carry aliasing error.",
                nlat,
                truncation,
                self.round_trip_error,
            )

    def _round_trip_error(self) -> float:
        """Relative error of re-analysing a random field band-limited to the truncation."""
        generator = torch.Generator().manual_seed(0)
        shape = (self.lmax, self.lmax)
        coeffs = torch.complex(torch.randn(shape, generator=generator), torch.randn(shape, generator=generator))
        coeffs[:, 0] = coeffs[:, 0].real.to(coeffs.dtype)  # m = 0 coefficients of a real field are real
        coeffs = torch.tril(coeffs)  # only m <= l exists
        with torch.no_grad():
            recovered = self.analyse(self.synthesise(coeffs))
        return float(torch.linalg.vector_norm(recovered - coeffs) / torch.linalg.vector_norm(coeffs))

    @property
    def lmax(self) -> int:
        """Number of degrees retained, ``l = 0 .. lmax-1``."""
        return self.truncation + 1

    def analyse(self, field: torch.Tensor) -> torch.Tensor:
        """Real field ``[..., points]`` to complex coefficients ``[..., lmax, lmax]``."""
        with torch.amp.autocast(device_type=field.device.type, enabled=False):
            return self._sht(field)

    def synthesise(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Complex coefficients ``[..., lmax, lmax]`` to a real field ``[..., points]``."""
        with torch.amp.autocast(device_type=coeffs.device.type, enabled=False):
            return self._isht(coeffs)

    def mean_square(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Area-weighted mean square of the field ``coeffs`` synthesise to, without synthesising it.

        Equal to the quadrature-weighted mean of ``synthesise(coeffs) ** 2`` (Parseval), at
        the cost of one pass over the coefficients instead of a transform.

        Parameters
        ----------
        coeffs : torch.Tensor
            Complex coefficients ``[..., lmax, lmax]`` of a real field.

        Returns
        -------
        torch.Tensor
            Real tensor of shape ``[...]``.
        """
        power = coeffs.real**2 + coeffs.imag**2
        return torch.einsum("...lm,lm->...", power, self.power_weight.to(power.dtype))

    def forward(self, field: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        """Filter ``field`` of shape ``[..., points]`` with a per-degree ``response``.

        ``response`` has shape ``[..., lmax]`` and broadcasts against the leading
        dimensions of ``field``, so ``[channels, lmax]`` filters each channel of a
        ``[..., channels, points]`` field with its own response.
        """
        return self.synthesise(self.analyse(field) * response.unsqueeze(-1))
