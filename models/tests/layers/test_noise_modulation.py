# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import pytest
import torch

from anemoi.models.layers.noise_modulation import GroupedStdModulation
from anemoi.models.layers.noise_modulation import resolve_prefixed_variables
from anemoi.models.layers.noise_modulation import strip_level
from anemoi.models.layers.spectral_helpers import quadrature_weights
from anemoi.models.layers.spectral_transforms import SphericalSpectralFilter
from anemoi.models.layers.spherical_noise import heat_kernel_response

NLAT = 16
LONS = [2 * NLAT] * NLAT
NUM_POINTS = sum(LONS)

# A model input mixing prognostics, forcings and prefixed spread fields, out of order.
NAME_TO_INDEX = {
    "t_850": 0,
    "std_t_850": 1,
    "std_2t": 2,
    "cos_latitude": 3,
    "std_u_500": 4,
    "std_10u": 5,
    "std_q_850": 6,
    "std_w_500": 7,
}
GROUPS = {"A": ["t", "2t"], "B": ["u", "10u"], "C": ["q"]}
CHANNEL_GROUP = ["A", "A", "B", "C"]
NUM_STD = 6


@pytest.fixture(scope="module")
def area_weights() -> torch.Tensor:
    return torch.as_tensor(quadrature_weights(LONS))


@pytest.fixture(scope="module")
def spectral_filter() -> SphericalSpectralFilter:
    return SphericalSpectralFilter(LONS, truncation=NLAT - 1)


def make_modulation(area_weights, **kwargs) -> GroupedStdModulation:
    params = {
        "groups": GROUPS,
        "channel_group": CHANNEL_GROUP,
        "name_to_index": NAME_TO_INDEX,
        "area_weights": area_weights,
    }
    params.update(kwargs)
    return GroupedStdModulation(**params)


def random_std(batch: int = 2, time: int = 2, seed: int = 0) -> torch.Tensor:
    """Positive, spatially varying spread fields, shape (batch, time, points, variables)."""
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(batch, time, NUM_POINTS, NUM_STD, generator=generator) * 3.0 + 0.1


@pytest.mark.parametrize(
    "name, base",
    [("t_850", "t"), ("2t", "2t"), ("10u", "10u"), ("q_1000", "q"), ("cos_latitude", "cos_latitude")],
)
def test_strip_level(name, base) -> None:
    assert strip_level(name) == base


def test_spread_variables_are_resolved_in_input_order() -> None:
    resolved = resolve_prefixed_variables(NAME_TO_INDEX, "std_")
    assert [index for index, _ in resolved] == [1, 2, 4, 5, 6, 7]


def test_groups_match_by_base_exact_or_full_name(area_weights) -> None:
    modulation = make_modulation(
        area_weights, groups={"A": ["t"], "B": ["u_500"], "C": ["std_q_850"]}, channel_group=["A", "B", "C"]
    )
    assert modulation.group_sizes() == {"A": 1, "B": 1, "C": 1}
    assert make_modulation(area_weights).group_sizes() == {"A": 2, "B": 2, "C": 1}


def test_ungrouped_spread_variables_are_reported(area_weights, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        make_modulation(area_weights)
    assert "['w']" in caplog.text


@pytest.mark.parametrize("smooth", [False, True])
def test_uniform_spread_gives_unit_amplitude(area_weights, spectral_filter, smooth) -> None:
    kwargs = {}
    if smooth:
        kT = torch.tensor([1e-3, 4e-3, 1e-3, 1.6e-2])
        kwargs = {"spectral_filter": spectral_filter, "smooth_response": heat_kernel_response(kT, NLAT)}
    modulation = make_modulation(area_weights, **kwargs)
    std = torch.ones(2, 2, NUM_POINTS, NUM_STD) * torch.tensor([0.1, 5.0, 2.0, 7.0, 1e-4, 3.0])

    amplitude = modulation(std)

    assert amplitude.shape == (2, 2, len(CHANNEL_GROUP), NUM_POINTS)
    torch.testing.assert_close(amplitude, torch.ones_like(amplitude), atol=1e-5, rtol=1e-5)


def test_amplitude_is_invariant_to_per_variable_scaling(area_weights) -> None:
    """Spread in kg/kg and in m^2/s^2 must contribute alike, whatever the data normaliser did."""
    modulation = make_modulation(area_weights)
    std = random_std()
    rescaled = std * torch.tensor([1e-5, 1.0, 300.0, 2.0, 1e-3, 9.0])

    torch.testing.assert_close(modulation(std), modulation(rescaled), atol=1e-5, rtol=1e-5)


def test_amplitude_has_unit_area_weighted_mean_square(area_weights) -> None:
    modulation = make_modulation(area_weights)
    amplitude = modulation(random_std())

    mean_square = modulation.area_mean(amplitude**2)
    torch.testing.assert_close(mean_square, torch.ones_like(mean_square))


def test_clip_bounds_the_amplitude_relative_to_its_mean(area_weights) -> None:
    modulation = make_modulation(area_weights, clip=(0.5, 2.0), preserve_total_variance=False)
    std = random_std()
    std[..., :100, :] *= 50.0  # an extreme patch

    amplitude = modulation(std)

    assert amplitude.min() >= 0.5
    assert amplitude.max() <= 2.0


def test_channels_follow_their_group(area_weights) -> None:
    modulation = make_modulation(area_weights)
    amplitude = modulation(random_std())

    torch.testing.assert_close(amplitude[:, :, 0], amplitude[:, :, 1])  # both group A, no smoothing
    assert not torch.allclose(amplitude[:, :, 0], amplitude[:, :, 2])


def test_group_map_is_the_mean_of_its_normalised_members(area_weights) -> None:
    modulation = make_modulation(area_weights, groups={"C": ["q"]}, channel_group=["C"], clip=None)
    std = random_std()
    q = std[..., 4]
    expected = q / modulation.area_mean(q).unsqueeze(-1)
    expected = expected / torch.sqrt(modulation.area_mean(expected**2)).unsqueeze(-1)

    torch.testing.assert_close(modulation(std)[:, :, 0], expected)


def test_smoothing_applies_each_channels_heat_kernel(area_weights, spectral_filter) -> None:
    """Smoothed map = raw group map with every degree scaled by that channel's heat kernel."""
    kT = torch.tensor([1e-3, 2e-2, 4e-3, 1.6e-2])
    response = heat_kernel_response(kT, NLAT).float()
    unclipped = {"clip": None, "preserve_total_variance": False}
    smoothed = make_modulation(area_weights, spectral_filter=spectral_filter, smooth_response=response, **unclipped)
    raw = make_modulation(area_weights, **unclipped)
    std = random_std()

    expected = spectral_filter.analyse(raw(std)) * response.unsqueeze(-1)

    torch.testing.assert_close(spectral_filter.analyse(smoothed(std)), expected, atol=1e-4, rtol=1e-4)
    # same group, different scale: the larger-kT channel is smoother
    assert smoothed(std)[:, :, 1].std() < smoothed(std)[:, :, 0].std()


def test_area_weighting_changes_the_normalisation(area_weights) -> None:
    uniform = torch.full((NUM_POINTS,), 1.0 / NUM_POINTS)
    std = random_std()
    std[..., : 2 * NLAT, :] *= 10.0  # high spread on the polar row only

    assert not torch.allclose(make_modulation(area_weights)(std), make_modulation(uniform)(std))


def test_non_positive_spread_is_rejected(area_weights) -> None:
    modulation = make_modulation(area_weights)
    with pytest.raises(ValueError, match="mean-std"):
        modulation(random_std() - 5.0)


def test_missing_spread_variables_are_rejected(area_weights) -> None:
    with pytest.raises(ValueError, match="No model input variable starts with"):
        make_modulation(area_weights, name_to_index={"t_850": 0, "cos_latitude": 1})


def test_group_matching_nothing_is_rejected(area_weights) -> None:
    with pytest.raises(ValueError, match="matches no"):
        make_modulation(area_weights, groups={**GROUPS, "D": ["tcw"]})


def test_undefined_channel_group_is_rejected(area_weights) -> None:
    with pytest.raises(ValueError, match="undefined groups"):
        make_modulation(area_weights, channel_group=["A", "Z"])


def test_invalid_clip_is_rejected(area_weights) -> None:
    with pytest.raises(ValueError, match="clip"):
        make_modulation(area_weights, clip=(4.0, 0.25))


def test_smoothing_needs_a_filter(area_weights) -> None:
    with pytest.raises(ValueError, match="spectral_filter"):
        make_modulation(area_weights, smooth_response=torch.ones(len(CHANNEL_GROUP), NLAT))


def test_nothing_is_checkpointed(area_weights, spectral_filter) -> None:
    kT = torch.full((len(CHANNEL_GROUP),), 1e-3)
    modulation = make_modulation(
        area_weights, spectral_filter=spectral_filter, smooth_response=heat_kernel_response(kT, NLAT)
    )
    assert modulation.state_dict() == {}
