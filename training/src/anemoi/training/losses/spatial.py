# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging

import torch
import torch.fft

from anemoi.training.losses.weightedloss import FunctionalWeightedLoss

LOGGER = logging.getLogger(__name__)


def amplitude(spectrum: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(spectrum.real**2 + spectrum.imag**2)


def get_power_spectra_scalar_product(
    power_spectra_real: torch.Tensor,
    power_spectra_fake: torch.Tensor,
) -> torch.Tensor:
    return power_spectra_real * power_spectra_fake.conj() + power_spectra_fake * power_spectra_real.conj()


def get_spectra(
    fake_output: torch.Tensor,
    real_output: torch.Tensor,
    dims: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    assert dims[0] * dims[1] == real_output.shape[2]
    dims_total = (*real_output.shape[:2], *dims, real_output.shape[-1])
    power_spectra_real = torch.fft.rfft2(real_output.reshape(dims_total), dim=(-2, -3))
    power_spectra_fake = torch.fft.rfft2(fake_output.reshape(dims_total), dim=(-2, -3))
    return power_spectra_real, power_spectra_fake


def log_rfft2_distance(fake_output: torch.Tensor, real_output: torch.Tensor, dims: tuple[int, int]) -> torch.Tensor:
    power_spectra_real, power_spectra_fake = get_spectra(fake_output, real_output, dims)
    epsilon = torch.finfo(real_output.dtype).eps  # Small epsilon to avoid division by zero
    power_spectra_real_sq = amplitude(power_spectra_real) ** 2
    power_spectra_fake_sq = amplitude(power_spectra_fake) ** 2
    ratio = (power_spectra_real_sq + epsilon) / (power_spectra_fake_sq + epsilon)

    def log10(x: torch.Tensor) -> torch.Tensor:
        return torch.log(x) / torch.log(torch.tensor(10.0, device=x.device, dtype=x.dtype))

    log_10 = (10 * log10(ratio)) ** 2
    return torch.sqrt(torch.mean(log_10)) / 10


class LogFFT2Distance(FunctionalWeightedLoss):
    r"""The log spectral distance is used to compute the difference between spectra of two fields.

    It is also called log spectral distorsion.
    When it is expressed in discrete space with L2 norm, it is defined as:
    <math>D_{LS}={\left\{ \frac{1}{N} \sum_{n=1}^N \left[ \log P(n) - \log\hat{P}(n)\right]^2\right\\}}^{1/2} ,</math>.
    All scaling and weighting is handled by the parent class.
    """

    def __init__(
        self,
        node_weights: torch.Tensor,
        ignore_nans: bool = False,
    ) -> None:
        super().__init__(node_weights, ignore_nans)

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return log_rfft2_distance(pred, target, dims=(self.x_dim, self.y_dim))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scalar_indices: tuple[int, ...] | None = None,
        without_scalars: list[str] | list[int] | None = None,
    ) -> torch.Tensor:
        result = super().forward(pred, target, squash, scalar_indices=scalar_indices, without_scalars=without_scalars)
        return torch.sqrt(torch.mean(result))


class FourierCorrelationLoss(FunctionalWeightedLoss):
    r"""The log spectral distance is used to compute the difference between spectra of two fields.

    See https://arxiv.org/pdf/2410.23159.pdf for more details.
    """

    def __init__(
        self,
        node_weights: torch.Tensor,
        x_dim: int,
        y_dim: int,
        time_weights: torch.Tensor = None,
        ignore_nans: bool = False,
    ) -> None:
        super().__init__(node_weights, time_weights, ignore_nans)
        self.x_dim = x_dim
        self.y_dim = y_dim

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        power_spectra_real, power_spectra_fake = get_spectra(pred, target, dims=(self.x_dim, self.y_dim))
        self.power_spectra_real = power_spectra_real
        self.power_spectra_fake = power_spectra_fake
        return get_power_spectra_scalar_product(power_spectra_real, power_spectra_fake).real

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scalar_indices: tuple[int, ...] | None = None,
        without_scalars: list[str] | list[int] | None = None,
    ) -> torch.Tensor:
        # scaling the scalar product with node weights
        scaled_scalar_product = super().forward(
            pred,
            target,
            squash,
            scalar_indices=scalar_indices,
            without_scalars=without_scalars,
        )
        # the rest of the loss implies summing over spatial dimensions
        numerator = (1 / 2) * torch.sum(
            scaled_scalar_product,
            dim=(-2, -3),
        )
        # and scaling with the spectrum amplitude: needs to be done after scaling with node weights
        # otherwise the spatial dimension is lost
        epsilon = torch.finfo(target.dtype).eps  # Small epsilon to avoid division by zero
        denominator = torch.sqrt(
            torch.sum(amplitude(self.power_spectra_real) ** 2, dim=(-2, -3))
            * torch.sum(amplitude(self.power_spectra_fake) ** 2, dim=(-2, -3))
            + epsilon,
        )
        return torch.mean(1 - numerator / denominator)
