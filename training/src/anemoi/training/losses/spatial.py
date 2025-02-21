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


def log_spectral_distance(real_output: torch.Tensor, fake_output: torch.Tensor) -> torch.Tensor:
    epsilon = torch.finfo(torch.float32).eps  # Small epsilon to avoid division by zero
    power_spectra_real = torch.abs(torch.fft.rfft2(real_output)) ** 2
    power_spectra_fake = torch.abs(torch.fft.rfft2(fake_output)) ** 2
    ratio = (power_spectra_real + epsilon) / (power_spectra_fake + epsilon)

    def log10(x: torch.Tensor) -> torch.Tensor:
        return torch.log(x) / torch.log(torch.tensor(10.0, device=x.device, dtype=x.dtype))

    return (10 * log10(ratio)) ** 2


class LogSpectralDistance(FunctionalWeightedLoss):
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
        return log_spectral_distance(pred, target)

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
