# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Float32 against float64 for the losses on spherical transforms, with variables of very different sizes."""

from collections.abc import Callable

import pytest
import torch

from anemoi.models.layers.spectral_helpers import InverseSphericalHarmonicTransform
from anemoi.models.layers.spectral_transforms import octahedral_lons_per_lat
from anemoi.training.losses import WaveletEnergyScoreLoss
from anemoi.training.losses.spectral import PowerSpectrumLoss
from anemoi.training.losses.spectral import SpectralCRPSLoss

NLAT = 24
TRUNCATION = 11
SMALL_VARIABLE_SIZES = [1.0, 1e-2, 1e-4, 1e-6]


def fields(members: int, seed: int, size: float) -> torch.Tensor:
    """Return random fields [member, grid, 2 variables] in float64, with the second variable of the given size."""
    generator = torch.Generator().manual_seed(seed)
    coefficients = torch.randn(members, 2, TRUNCATION + 1, TRUNCATION + 1, dtype=torch.complex128, generator=generator)
    coefficients = coefficients.tril()
    coefficients[..., 0] = coefficients[..., 0].real
    inverse = InverseSphericalHarmonicTransform(octahedral_lons_per_lat(NLAT), TRUNCATION)
    grid = inverse(coefficients).movedim(1, -1)
    return grid * torch.tensor([1.0, size], dtype=torch.float64)


def spectral_crps(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    loss = SpectralCRPSLoss(transform="octahedral_sht", nlat=NLAT)
    pred = pred[None, None].clone().requires_grad_()
    value = loss(pred, target[None, None])
    value.backward()
    return value.detach(), pred.grad[0, 0]


def wavelet_score(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    loss = WaveletEnergyScoreLoss(
        transform="octahedral_sht",
        nlat=NLAT,
        scale_weights=[0.0, 0.0, 1.0, 2.0, 0.0],
        smoothing_octaves=2,
    ).to(pred.dtype)
    structure = loss.structure
    members = pred.movedim(-1, 0).clone().requires_grad_()  # [variable, member, grid]
    target_coefficients = structure.coefficients(target.movedim(-1, 0))
    coefficients = structure.coefficients(members)
    scores = 0.0
    for k, weight in enumerate(loss.weights):
        vectors = loss.structure_vectors(structure.tensor(coefficients, k))
        target_vectors = loss.structure_vectors(structure.tensor(target_coefficients, k))
        scores = scores + weight * loss.energy_score(vectors, target_vectors)
    scores.sum().backward()
    return scores.detach().sum(-1), members.grad.movedim(0, -1)


@pytest.mark.parametrize("size", SMALL_VARIABLE_SIZES)
@pytest.mark.parametrize("loss", [spectral_crps, wavelet_score])
def test_small_variable_keeps_its_gradient_next_to_a_large_one(loss: Callable, size: float) -> None:
    pred, target = fields(3, 0, size), fields(1, 1, size)
    _, exact = loss(pred, target)
    _, actual = loss(pred.float(), target.float())
    error = (actual[..., 1].double() - exact[..., 1]).norm() / exact[..., 1].norm()
    assert error < 5e-6


@pytest.mark.parametrize("size", SMALL_VARIABLE_SIZES)
def test_wavelet_score_of_a_small_variable_keeps_its_accuracy(size: float) -> None:
    pred, target = fields(3, 0, size), fields(1, 1, size)
    exact, _ = wavelet_score(pred, target)
    actual, _ = wavelet_score(pred.float(), target.float())
    assert ((actual[1].double() - exact[1]) / exact[1]).abs() < 5e-6


def power_spectrum(pred: torch.Tensor, target: torch.Tensor, autocast: bool) -> tuple[torch.Tensor, torch.Tensor]:
    loss = PowerSpectrumLoss(transform="octahedral_sht", nlat=NLAT)
    pred = pred[None, None].clone().requires_grad_()
    # As in training, the forward pass runs under autocast and the backward pass after it.
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        value = loss(pred, target[None, None])
    value.backward()
    return value.detach(), pred.grad[0, 0]


def test_power_spectrum_keeps_float32_under_bfloat16_autocast() -> None:
    """The loss leaves autocast on, and its transforms still run in float32."""
    pred, target = fields(1, 0, 1.0), fields(1, 1, 1.0)
    exact_value, exact_gradient = power_spectrum(pred, target, autocast=False)
    value, gradient = power_spectrum(pred.float(), target.float(), autocast=True)
    assert ((value.double() - exact_value) / exact_value).abs() < 1e-5
    assert (gradient.double() - exact_gradient).norm() / exact_gradient.norm() < 1e-5


def test_wavelet_loss_keeps_float64_fields_in_float64() -> None:
    pred, target = fields(3, 0, 1e-2), fields(1, 1, 1e-2)
    loss = WaveletEnergyScoreLoss(
        transform="octahedral_sht",
        nlat=NLAT,
        scale_weights=[0.0, 0.0, 1.0, 2.0, 0.0],
        smoothing_octaves=2,
    )
    scores = loss.point_scores(pred[None, None], target[None, None])
    assert scores.dtype == torch.float64
    exact, _ = wavelet_score(pred, target)
    torch.testing.assert_close(scores[0, 0, 0].sum(0), exact, rtol=1e-10, atol=0)
