# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Spectral low-pass scales for the multiscale loss."""

from abc import ABC
from abc import abstractmethod

import torch

from anemoi.models.layers.spectral_transforms import DCT2D
from anemoi.models.layers.spectral_transforms import FFT2D
from anemoi.models.layers.spectral_transforms import InverseDCT2D
from anemoi.models.layers.spectral_transforms import InverseFFT2D
from anemoi.models.layers.spectral_transforms import InverseOctahedralSHT
from anemoi.models.layers.spectral_transforms import InverseReducedSHT
from anemoi.models.layers.spectral_transforms import InverseRegularSHT
from anemoi.models.layers.spectral_transforms import InverseSpectralTransform
from anemoi.models.layers.spectral_transforms import OctahedralSHT
from anemoi.models.layers.spectral_transforms import ReducedSHT
from anemoi.models.layers.spectral_transforms import RegularSHT
from anemoi.models.layers.spectral_transforms import SpectralTransform

SPHERICAL_TRANSFORMS = {
    "octahedral_sht": (OctahedralSHT, InverseOctahedralSHT),
    "reduced_sht": (ReducedSHT, InverseReducedSHT),
    "regular_sht": (RegularSHT, InverseRegularSHT),
}
PLANAR_TRANSFORMS = {
    "fft2d": (FFT2D, InverseFFT2D),
    "dct2d": (DCT2D, InverseDCT2D),
}


class SpectralScales(torch.nn.Module, ABC):
    """Splits fields into spectral scales that share one forward transform.

    A field is analysed once. Each scale is then built back on the grid from the
    coefficients up to its cutoff. The transforms always run in float32, because
    complex tensors have no bfloat16 form.
    """

    def __init__(self, transform: SpectralTransform) -> None:
        super().__init__()
        self.transform = transform

    def analyse(self, x: torch.Tensor) -> torch.Tensor:
        """Turn ``[batch, time, ensemble, grid, variables]`` into coefficients ``[..., variables, k_y, k_x]``."""
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            return self.transform(x.float()).movedim(-1, -3)

    def synthesise(self, coeffs: torch.Tensor, cutoff: float) -> torch.Tensor:
        """Build the grid field ``[..., grid, variables]`` from the coefficients up to ``cutoff``."""
        with torch.amp.autocast(device_type=coeffs.device.type, enabled=False):
            return self._synthesise(coeffs, cutoff).transpose(-1, -2)

    @abstractmethod
    def _synthesise(self, coeffs: torch.Tensor, cutoff: float) -> torch.Tensor:
        """Build ``[..., variables, grid]`` from the coefficients up to ``cutoff``."""


class SphericalHarmonicScales(SpectralScales):
    """Scales cut at total wavenumbers on the sphere.

    The forward transform runs up to the finest truncation. The coefficients of a
    coarser scale are the leading rows and columns of that analysis, so each scale
    only runs an inverse transform sized to its own truncation.
    """

    def __init__(self, transform: SpectralTransform, inverses: dict[int, InverseSpectralTransform]) -> None:
        super().__init__(transform)
        self.inverses = torch.nn.ModuleDict({str(truncation): inverse for truncation, inverse in inverses.items()})

    def _synthesise(self, coeffs: torch.Tensor, cutoff: int) -> torch.Tensor:
        return self.inverses[str(cutoff)](coeffs[..., : cutoff + 1, : cutoff + 1])


class PlanarScales(SpectralScales):
    """Scales cut at a radial frequency on a rectangular grid.

    ``frequency`` gives, for every coefficient, its frequency in cycles per grid
    spacing; a scale keeps the coefficients at or below its cutoff. The grid
    spacing is taken to be the same in x and y.
    """

    def __init__(
        self,
        transform: SpectralTransform,
        inverse: InverseSpectralTransform,
        frequency: torch.Tensor,
    ) -> None:
        super().__init__(transform)
        self.inverse = inverse
        self.register_buffer("frequency", frequency, persistent=False)

    def _synthesise(self, coeffs: torch.Tensor, cutoff: float) -> torch.Tensor:
        return self.inverse(coeffs * (self.frequency <= cutoff))


def _radial_frequency(frequency_y: torch.Tensor, frequency_x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(frequency_y[:, None] ** 2 + frequency_x[None, :] ** 2)


def build_spectral_scales(transform: str, cutoffs: list[float], **grid_kwargs) -> SpectralScales:
    """Build the spectral scales for the named transform.

    Parameters
    ----------
    transform : str
        One of the spherical harmonic transforms ("octahedral_sht", "reduced_sht",
        "regular_sht") or the rectangular-grid transforms ("fft2d", "dct2d").
    cutoffs : list[float]
        Strictly increasing cutoffs, one per scale. For spherical harmonic
        transforms these are integer truncations; for rectangular grids they are
        frequencies in cycles per grid spacing, so a cutoff of 0.125 keeps
        wavelengths of 8 grid spacings and longer.
    grid_kwargs
        Grid description passed to the transforms: ``nlat`` or ``grid`` for the
        spherical harmonic transforms, ``x_dim`` and ``y_dim`` for rectangular grids.

    Returns
    -------
    SpectralScales
        The shared forward transform with what each scale needs to be built back on the grid.
    """
    if transform in SPHERICAL_TRANSFORMS:
        if not all(isinstance(cutoff, int) for cutoff in cutoffs):
            msg = f"Spherical harmonic cutoffs must be integer truncations, got {cutoffs}."
            raise ValueError(msg)
        forward_cls, inverse_cls = SPHERICAL_TRANSFORMS[transform]
        return SphericalHarmonicScales(
            forward_cls(truncation=cutoffs[-1], **grid_kwargs),
            {cutoff: inverse_cls(truncation=cutoff, **grid_kwargs) for cutoff in cutoffs},
        )

    if transform in PLANAR_TRANSFORMS:
        forward_cls, inverse_cls = PLANAR_TRANSFORMS[transform]
        x_dim, y_dim = grid_kwargs["x_dim"], grid_kwargs["y_dim"]
        if transform == "fft2d":
            frequency = _radial_frequency(torch.fft.fftfreq(y_dim), torch.fft.fftfreq(x_dim))
        else:
            # Cosine k spans k half-waves across the domain.
            frequency = _radial_frequency(torch.arange(y_dim) / (2 * y_dim), torch.arange(x_dim) / (2 * x_dim))
        return PlanarScales(forward_cls(**grid_kwargs), inverse_cls(**grid_kwargs), frequency)

    msg = (
        f"Unknown multiscale transform: {transform}. "
        f"Use one of {sorted(SPHERICAL_TRANSFORMS) + sorted(PLANAR_TRANSFORMS)}."
    )
    raise ValueError(msg)
