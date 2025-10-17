# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import torch
import torch.fft

from anemoi.training.losses.base import FunctionalLoss
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


def amplitude(spectrum: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(spectrum.real**2 + spectrum.imag**2)


def get_power_spectra_scalar_product(
    power_spectra_real: torch.Tensor,
    power_spectra_pred: torch.Tensor,
) -> torch.Tensor:
    return power_spectra_real * power_spectra_pred.conj() + power_spectra_pred * power_spectra_real.conj()


def get_spectra(
    predicted_output: torch.Tensor,
    real_output: torch.Tensor,
    dims: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    assert dims[0] * dims[1] == real_output.shape[TensorDim.GRID], (
        "The product of dims must match the spatial dims of the output."
        "Please use x_dim and y_dim such that field_shape=(x_dim, y_dim)."
    )
    dims_total = (*real_output.shape[: TensorDim.GRID], *dims, real_output.shape[TensorDim.VARIABLE])
    power_spectra_real = torch.fft.fft2(real_output.reshape(dims_total), dim=(-2, -3))
    power_spectra_pred = torch.fft.fft2(predicted_output.reshape(dims_total), dim=(-2, -3))
    return power_spectra_real, power_spectra_pred


def log_rfft2_distance(
    predicted_output: torch.Tensor,
    real_output: torch.Tensor,
    dims: tuple[int, int],
) -> torch.Tensor:
    r"""Calculate the log spectral distance between two fields."""
    power_spectra_real, power_spectra_pred = get_spectra(predicted_output, real_output, dims)
    epsilon = torch.finfo(real_output.dtype).eps  # Small epsilon to avoid division by zero
    power_spectra_real_sq = amplitude(power_spectra_real) ** 2
    power_spectra_pred_sq = amplitude(power_spectra_pred) ** 2
    ratio = (power_spectra_real_sq + epsilon) / (power_spectra_pred_sq + epsilon)

    def log10(x: torch.Tensor) -> torch.Tensor:
        return torch.log(x) / torch.log(torch.tensor(10.0, device=x.device, dtype=x.dtype))

    return (10 * log10(ratio)) ** 2


class LogFFT2Distance(FunctionalLoss):
    r"""The log spectral distance is used to compute the difference between spectra of two fields.

    It is also called log spectral distortion.
    When it is expressed in discrete space with L2 norm, it is defined as:
    <math>D_{LS}={\left\{ \frac{1}{N} \sum_{n=1}^N \left[ \log P(n) - \log\hat{P}(n)\right]^2\right\\}}^{1/2} ,</math>.
    All scaling and weighting is handled by the parent class.
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        ignore_nans: bool = False,
    ) -> None:
        super().__init__(ignore_nans)
        LOGGER.warning(
            "LogFFT2Distance can only be used with data on 2D grids.",
        )
        self.x_dim = x_dim
        self.y_dim = y_dim

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dist = log_rfft2_distance(pred, target, dims=(self.x_dim, self.y_dim))
        return dist.reshape(pred.shape)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
    ) -> torch.Tensor:
        result = super().forward(pred, target, squash, scaler_indices=scaler_indices, without_scalers=without_scalers)
        return torch.sqrt(torch.mean(result))


class FourierCorrelationLoss(FunctionalLoss):
    r"""The log spectral distance is used to compute the difference between spectra of two fields.

    See https://arxiv.org/pdf/2410.23159.pdf for more details.
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        ignore_nans: bool = False,
    ) -> None:
        super().__init__(ignore_nans)
        LOGGER.warning(
            "Fourier Correlation loss can only be used with data on 2D grids.",
        )
        self.x_dim = x_dim
        self.y_dim = y_dim

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        power_spectra_real, power_spectra_pred = get_spectra(pred, target, dims=(self.x_dim, self.y_dim))
        self.power_spectra_real = power_spectra_real
        self.power_spectra_pred = power_spectra_pred
        dist = get_power_spectra_scalar_product(power_spectra_real, power_spectra_pred).real
        return dist.reshape(pred.shape)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
    ) -> torch.Tensor:
        # scaling the scaler product with node weights
        scaled_scalar_product = super().forward(
            pred,
            target,
            squash,
            scaler_indices=scaler_indices,
            without_scalers=without_scalers,
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
            * torch.sum(amplitude(self.power_spectra_pred) ** 2, dim=(-2, -3))
            + epsilon,
        )
        return torch.mean(1 - numerator / denominator)
