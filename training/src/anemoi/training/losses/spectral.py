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
from typing import Literal

import einops
import torch
import torch.fft
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import FunctionalLoss

LOGGER = logging.getLogger(__name__)


class SpectralTransform:

    @abc.abstractmethod
    def __call__(
        self,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """Transform data to spectral domain.

        Parameters
        ----------
        data : torch.Tensor
            Input data in the spatial domain.

        Returns
        -------
        torch.Tensor
            Data transformed to the spectral domain.
        """


class FFT2D(SpectralTransform):

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
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

    def __call__(
        self,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """Transform data to spectral domain using 2D FFT.

        Parameters
        ----------
        data : torch.Tensor
            Input data in the spatial domain.

        Returns
        -------
        torch.Tensor
            Data transformed to the spectral domain.
        """
        batch_size, time, _, var = data.shape
        # [batch, time, y*x, variables] -> [batch*time*variables, y, x]
        data = einops.rearrange(data, "b t (y x) v -> (b t v) y x", x=self.x_dim, y=self.y_dim)
        fft_data = torch.fft.fft2(data)
        # [batch*time*variables, y, x] -> [batch, time, y*x, variables]
        return einops.rearrange(fft_data, "(b t v) y x -> b t (y x) v", b=batch_size, t=time, v=var)


class SHT(SpectralTransform):
    """Placeholder for Spherical Harmonics Transform."""

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
        msg = "Spherical harmonics transform is not implemented yet."
        raise NotImplementedError(msg)


class SpectralLoss(BaseLoss):
    """Base class for spectral losses."""

    def __init__(
        self,
        transform: Literal["fft2d", "sht"],
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        """Spectral Loss Base.

        Parameters
        ----------
        ignore_nans : bool
            whether to ignore NaNs in the loss computation
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
            + 1e-12,
        )

        return 1 - numerator / denominator
