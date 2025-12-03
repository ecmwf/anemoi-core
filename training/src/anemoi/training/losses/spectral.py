# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Literal

import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.layers.spectral_transforms import FFT2D
from anemoi.models.layers.spectral_transforms import SHT
from anemoi.models.layers.spectral_transforms import SpectralTransform
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import FunctionalLoss

LOGGER = logging.getLogger(__name__)


class SpectralLoss(BaseLoss):
    """Base class for spectral losses."""

    transform: SpectralTransform

    def __init__(
        self,
        transform: Literal["fft2d", "sht"],
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        """Spectral Loss Base.

        Parameters
        ----------
        transform : Literal["fft2d", "sht"]
            type of spectral transform to use
        ignore_nans : bool
            whether to ignore NaNs in the loss computation
        kwargs : dict
            additional arguments for the spectral transform
        """
        super().__init__(ignore_nans, **kwargs)
        if transform == "fft2d":
            self.transform = FFT2D(**kwargs)
        elif transform == "sht":
            self.transform = SHT()
        else:
            msg = f"Unknown transform type: {transform}"
            raise ValueError(msg)


class FunctionalSpectralLoss(FunctionalLoss, SpectralLoss):
    """Base class for functional spectral losses.

    Combines spectral transformation with functional loss computation,
    by simply transforming the inputs before passing them to the functional loss.
    """

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the difference between predicted and target in spectral domain."""
        return super().calculate_difference(pred, target)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the functional spectral loss between predicted and target data."""
        pred_spectral = self.transform(pred)
        target_spectral = self.transform(target)
        return super().forward(pred_spectral, target_spectral, without_scalers=-2, **kwargs)


class SpectralL2Loss(FunctionalSpectralLoss):
    r"""Standard Fourier-domain loss.

    Implements a loss based on the difference in the spectral domain, expressed as:
    .. math::
        \mathrm{FourierLoss}(X, \hat X)
            = \lVert F - \hat F \rVert_p^2
    By default uses L2 loss in spectral domain.
    """

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target) ** 2


class LogSpectralDistance(SpectralLoss):
    r"""Log Spectral Distance (LSD).

    The log spectral distance is used to compute the difference between spectra of two fields.
    It is also called log spectral distortion. When it is expressed in discrete space with L2 norm, it is defined as:
    .. math::
        D_{LS}={\left\{ \frac{1}{N} \sum_{n=1}^N \left[ \log P(n) - \log\hat{P}(n)\right]^2\right\\}}^{1/2} ,
    where P(n) and ^P(n) are the power spectral densities of the true and predicted fields respectively.
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = -2,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
    ) -> torch.Tensor:

        eps = torch.finfo(pred.dtype).eps

        pred_spectral = self.transform(pred)
        target_spectral = self.transform(target)

        power_spectra_real = torch.abs(pred_spectral) ** 2
        power_spectra_pred = torch.abs(target_spectral) ** 2

        log_diff = torch.log(power_spectra_real + eps) - torch.log(power_spectra_pred + eps)

        result = self.scale(
            log_diff**2,
            scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )

        return torch.sqrt(self.reduce(result, squash=squash, group=group) + eps)


class FourierCorrelationLoss(SpectralLoss):
    r"""Fourier Correlation Loss (FCL).

    Implements the loss proposed in [1]_ and expressed as:

    .. math::
        \mathrm{FCL}(X, \hat X)
            = 1 - \frac{1}{2} \; \frac{ P\bigl[F \,\hat F^* + F^*\,\hat F\bigr] }
                                    {\, \sqrt{ P\lvert F\rvert^2 \; P\lvert \hat F \rvert^2 }\, }

    References
    ----------
    .. [1] Yan, C.-W. et al. (2024).
        Fourier Amplitude and Correlation Loss: Beyond Using L2 Loss
        for Skillful Precipitation Nowcasting. arXiv:2410.23159.
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = -2,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
    ) -> torch.Tensor:

        eps = torch.finfo(pred.dtype).eps

        # transform to spectral domain
        pred_spectral = self.transform(pred)
        target_spectral = self.transform(target)

        # compute the cross power spectrum and numerator
        cross_power_spectrum = torch.real(pred_spectral * torch.conj(target_spectral))
        cross_power_spectrum = self.scale(
            cross_power_spectrum,
            scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )
        numerator = self.reduce(cross_power_spectrum, squash=squash, group=group)

        # compute the normalization using the amplitudes
        denominator = torch.sqrt(
            self.reduce(torch.abs(pred_spectral) ** 2, squash=squash, group=group)
            * self.reduce(torch.abs(target_spectral) ** 2, squash=squash, group=group)
            + eps,
        )

        return 1 - numerator / denominator
