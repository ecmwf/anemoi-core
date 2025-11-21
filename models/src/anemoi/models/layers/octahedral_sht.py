import logging
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from torch.fft import rfft

polytype = np.float64

LOGGER = logging.getLogger(__name__)

complex_dtype_map = {torch.float16: torch.complex32, torch.float32: torch.complex64, torch.float64: torch.complex128}


class EcTransOctahedralSHT:
    def __init__(self, truncation: int, dtype, filepath=None) -> None:
        self.truncation = truncation
        self.n_lat_nh = truncation + 1
        self.dtype = dtype
        self.n_lon_for_each_lat_nh = np.array([20 + 4 * i for i in range(self.n_lat_nh)])
        self.highest_zonal_wavenumber_per_lat_nh = None
        self.n_points = 2 * int(sum(self.n_lon_for_each_lat_nh))
        self.latitude_indices = list(range(self.n_lat_nh)) + list(range(self.n_lat_nh - 1, -1, -1))
        self.lons_per_lat = [20 + 4 * i for i in range(truncation + 1)]
        self.lons_per_lat += self.lons_per_lat[::-1]
        self.cumsum_indices = [0] + np.cumsum(self.lons_per_lat).tolist()

        self.gaussian_weights = None
        self.symmetric = []
        self.antisymmetric = []

        self._get_polynomials_and_weights(filepath)

        self.symmetric *= self.gaussian_weights.view(1, 1, -1)
        self.antisymmetric *= self.gaussian_weights.view(1, 1, -1)

        padding = [self.highest_zonal_wavenumber_per_lat_nh[-1] - m for m in self.highest_zonal_wavenumber_per_lat_nh]
        self.padding = padding + padding[::-1]

        self.highest_zonal_wavenumber_per_lat_nh = torch.from_numpy(self.highest_zonal_wavenumber_per_lat_nh)

    def _allocate_arrays_for_polynomials(self) -> None:
        # Allocate arrays for storing Legendre polynomials, antisymmetric and symmetric
        n_values_symm = sum(
            [self.n_lats_per_wavenumber[m] * (self.truncation - m + 3) // 2 for m in range(self.truncation + 1)]
        )
        legpol_symm = np.zeros(n_values_symm, dtype=polytype)
        n_values_anti = sum(
            [self.n_lats_per_wavenumber[m] * (self.truncation - m + 2) // 2 for m in range(self.truncation + 1)]
        )
        legpol_anti = np.zeros(n_values_anti, dtype=polytype)
        return legpol_symm, legpol_anti

    @cached_property
    def n_lats_per_wavenumber(self) -> list[int]:
        # Calculate latitudes involved in Legendre transform for each zonal wavenumber m, based on nmen
        assert self.highest_zonal_wavenumber_per_lat_nh is not None

        n_lats_per_wavenumber = np.zeros(self.truncation + 1, dtype=np.int32)
        for i in range(self.truncation + 1):
            n_lats_per_wavenumber[i] = self.highest_zonal_wavenumber_per_lat_nh[
                self.highest_zonal_wavenumber_per_lat_nh >= i
            ].shape[0]
        return n_lats_per_wavenumber

    @cached_property
    def poly_size(self) -> int:
        return sum(self.truncation + 2 - im for im in range(self.truncation + 1))

    def gererate(self):
        # Fetch relevant arrays from ecTrans
        # Note that all of these arrays (including the input points-per-latitude array) are
        # specified across the full globe, pole to pole

        import ectrans4py

        (highest_zonal_wavenumber_per_lat, gaussian_weights, all_legendre_polynomials) = ectrans4py.get_legendre_assets(
            2 * self.n_lat_nh,
            self.truncation,
            2 * self.n_lat_nh,
            self.poly_size,
            np.concat((self.n_lon_for_each_lat_nh, self.n_lon_for_each_lat_nh[::-1])),
            1,
        )
        return highest_zonal_wavenumber_per_lat, gaussian_weights, all_legendre_polynomials

    def generate_and_save(self, filepath: Path):
        highest_zonal_wavenumber_per_lat, gaussian_weights, legendre_polynomials = self.gererate()
        if filepath:
            np.savez(
                filepath,
                legendre_polynomials=legendre_polynomials,
                gaussian_weights=gaussian_weights,
                highest_zonal_wavenumber_per_lat=highest_zonal_wavenumber_per_lat,
            )
        return highest_zonal_wavenumber_per_lat, gaussian_weights, legendre_polynomials

    def load_from_disk(self, filepath: Path):
        loaded_assets = np.load(filepath)
        return (
            loaded_assets["highest_zonal_wavenumber_per_lat"],
            loaded_assets["gaussian_weights"],
            loaded_assets["legendre_polynomials"],
        )

    def _get_polynomials_and_weights(self, filepath=None) -> None:

        if filepath and filepath.exists():
            self.highest_zonal_wavenumber_per_lat, gaussian_weights, all_legendre_polynomials = self.load_from_disk(
                filepath
            )
        else:
            self.highest_zonal_wavenumber_per_lat, gaussian_weights, all_legendre_polynomials = self.generate_and_save(
                filepath
            )

        # Flatten Legendre polynomial array to make it easier to unpack
        all_legendre_polynomials = all_legendre_polynomials.flatten()

        # Extract just the Northern hemisphere for these arrays
        self.highest_zonal_wavenumber_per_lat_nh = self.highest_zonal_wavenumber_per_lat[: self.n_lat_nh]

        gaussian_weights = gaussian_weights[: self.n_lat_nh]
        self.gaussian_weights = torch.from_numpy(gaussian_weights).to(self.dtype)

        # Read Legendre polynomials, looping over each zonal wavenumber m
        m_off_symm, m_off_anti = 0, 0
        off_symm, off_anti = 0, 0
        for m in range(self.truncation + 1):
            i_s = (m + 1) % 2
            i_a = m % 2

            n_lats = int(self.n_lats_per_wavenumber[m])
            n_total_values = self.truncation - m + 2  # Number of total wavenumbers valid at this m

            # Determine the maximum total wavenumber for this m, symm and antisymmetric
            max_total_wavenumber_symm = (self.truncation - m + 3) // 2
            max_total_wavenumber_anti = (self.truncation - m + 2) // 2

            symm_matrix = np.zeros(((self.truncation + 3) // 2, self.n_lat_nh), dtype=polytype)
            offset = (self.truncation + 3) // 2 - max_total_wavenumber_symm

            # Parse symmetric Legendre polynomials
            for i in range(max_total_wavenumber_symm):
                # Construct slicing indices for unpacked and packed arrays

                # In the packed array data for each (zonal, total) wavenumber are zero padded up to
                # Hence we have to add an extra offset to step through the zeroed phony latitudes
                all_1 = m_off_symm + 2 * i * self.n_lat_nh + (self.n_lat_nh - n_lats)
                all_2 = m_off_symm + (2 * i + 1) * self.n_lat_nh

                symm_matrix[i + offset, self.n_lat_nh - n_lats :] = all_legendre_polynomials[all_1:all_2]

            if i_s == 1:
                symm_matrix[offset, :] = 0

            self.symmetric.append(torch.from_numpy(symm_matrix))

            off_symm += max_total_wavenumber_symm * n_lats

            # Parse antisymmetric Legendre polynomials
            anti_matrix = np.zeros(((self.truncation + 2) // 2, self.n_lat_nh), dtype=polytype)

            offset = (self.truncation + 2) // 2 - max_total_wavenumber_anti
            for i in range(max_total_wavenumber_anti):

                # Ditto comment above for zero padding in the packed array
                all_1 = m_off_anti + (2 * i + 1) * self.n_lat_nh + (self.n_lat_nh - n_lats)
                all_2 = m_off_anti + (2 * i + 2) * self.n_lat_nh

                # Copy latitudes for this (zonal, total) wavenumber from packed to unpacked array
                anti_matrix[i + offset, self.n_lat_nh - n_lats :] = all_legendre_polynomials[all_1:all_2]

            if i_a == 1:
                anti_matrix[offset, :] = 0

            self.antisymmetric.append(torch.from_numpy(anti_matrix))

            off_anti += max_total_wavenumber_anti * n_lats

            # Offset into flattened Legendre polynomial array
            if m % 2 == 0:
                m_off_symm += (n_total_values + 1) * self.n_lat_nh
                m_off_anti += (n_total_values - 1) * self.n_lat_nh
            else:
                m_off_symm += (n_total_values - 1) * self.n_lat_nh
                m_off_anti += (n_total_values + 1) * self.n_lat_nh

        self.symmetric = torch.stack(self.symmetric).to(complex_dtype_map[self.dtype])
        self.antisymmetric = torch.stack(self.antisymmetric).to(complex_dtype_map[self.dtype])

    def longitudinal_rfft(self, x: torch.Tensor):
        four_out = []
        for i in range(2 * self.truncation + 2):

            out = rfft(x[:, :, self.cumsum_indices[i] : self.cumsum_indices[i + 1], :], axis=2, norm="forward")[
                :, :, : self.highest_zonal_wavenumber_per_lat[i] + 1, :
            ]
            four_out.append(
                torch.cat(
                    [out, torch.zeros((*out.shape[:2], self.padding[i], out.shape[-1]), device=out.device)], dim=2
                )
            )

        return torch.stack(four_out, dim=2)

    def legendre_quadrature(self, x: torch.Tensor):

        fourier_sh_flipped = torch.flip(x[:, :, self.n_lat_nh :, :, :], dims=[2])
        fourier_norm_sym = x[:, :, : self.n_lat_nh, :, :] + fourier_sh_flipped
        fourier_norm_anti = x[:, :, : self.n_lat_nh, :, :] - fourier_sh_flipped

        spectrum_sym = torch.einsum(
            "mnijk,jli->mnljk", fourier_norm_sym, self.symmetric.to(device=x.device)
        )  # noqa: F841
        spectrum_anti = torch.einsum(
            "mnijk,jli->mnljk", fourier_norm_anti, self.antisymmetric.to(device=x.device)
        )  # noqa: F841

        return (spectrum_sym, spectrum_anti)

    def __call__(self, x: torch.Tensor):

        x = self.longitudinal_rfft(x)
        x_sym, x_anti = self.legendre_quadrature(x)
        return x_sym, x_anti
