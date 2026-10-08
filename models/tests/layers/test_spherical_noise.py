# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math

import pytest
import torch

from anemoi.models.layers.spectral_helpers import quadrature_weights
from anemoi.models.layers.spectral_transforms import RegularSHT
from anemoi.models.layers.spherical_noise import DiffusionNoiseS2
from anemoi.models.layers.spherical_noise import DummyNoiseS2
from anemoi.models.layers.spherical_noise import band_limit
from anemoi.models.layers.spherical_noise import build_inverse_sht
from anemoi.models.layers.spherical_noise import build_noise
from anemoi.models.layers.spherical_noise import degree_variance
from anemoi.models.layers.spherical_noise import diffusion_band_limit
from anemoi.models.layers.spherical_noise import diffusion_degree_variance
from anemoi.models.layers.spherical_noise import diffusion_lowpass_response
from anemoi.models.layers.spherical_noise import filter_truncation
from anemoi.models.layers.spherical_noise import heat_kernel_response
from anemoi.models.layers.spherical_noise import lowpass_response
from anemoi.models.layers.spherical_noise import noise_seeds_reflects
from anemoi.models.layers.spherical_noise import orthonormal_basis_scale
from anemoi.models.layers.spherical_noise import per_channel_values
from anemoi.models.layers.spherical_noise import tabulated_coefficient_variance

NLAT = 32
ENSEMBLE = 4


@pytest.fixture(scope="module")
def transform() -> tuple:
    return build_inverse_sht(NLAT)


def make_noise(transform, params, *, batch_size=2, ensemble_size=ENSEMBLE, num_time_steps=1, centered=False, **kwargs):
    seeds, reflects = noise_seeds_reflects(ensemble_size, centered=centered, **kwargs)
    return build_noise(
        params,
        transform=transform,
        batch_size=batch_size,
        ensemble_size=ensemble_size,
        num_channels=1,
        num_time_steps=num_time_steps,
        seeds=seeds,
        reflects=reflects,
    )


def collect(noise, n_draws: int, *, replace_first: bool = True) -> torch.Tensor:
    """Draw ``n_draws`` successive fields, stacked along a leading axis."""
    fields = []
    for step in range(n_draws):
        noise.update(replace_state=(replace_first and step == 0))
        fields.append(noise().clone())
    return torch.stack(fields)


def test_basis_scale_is_four_pi_normalisation(transform) -> None:
    """Anemoi's legpoly uses the ECMWF 4-pi convention; torch-harmonics is orthonormal."""
    isht, _, _ = transform
    assert orthonormal_basis_scale(isht) == pytest.approx(math.sqrt(4 * math.pi), rel=1e-5)


def test_build_inverse_sht_defaults_lmax_to_nlat(transform) -> None:
    isht, lmax, num_grid_points = transform
    assert lmax == NLAT
    assert isht._isht.truncation == NLAT - 1
    assert num_grid_points == NLAT * 2 * NLAT


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_white_pointwise_std_matches_sigma(transform, sigma) -> None:
    """The Z normalisation is supposed to pin pointwise variance to sigma^2."""
    noise = make_noise(transform, {"type": "white", "sigma": sigma, "alpha": 0.0})
    fields = collect(noise, 40)
    assert fields.std().item() == pytest.approx(sigma, rel=0.05)
    assert fields.mean().item() == pytest.approx(0.0, abs=0.05 * sigma)


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_diffusion_pointwise_std_matches_sigma(transform, sigma) -> None:
    """The sqrt(1-phi^2) prefactor keeps the OU process stationary at sigma^2."""
    noise = make_noise(transform, {"type": "diffusion", "sigma": sigma, "kT": 0.00308057, "lambd": 1.0})
    fields = collect(noise, 60)
    assert fields.std().item() == pytest.approx(sigma, rel=0.05)


def test_diffusion_stationary_from_the_first_step(transform) -> None:
    """replace_state=True must sample the stationary distribution, not spin up to it."""
    noise = make_noise(transform, {"type": "diffusion", "sigma": 1.0, "lambd": 0.1})
    first = torch.stack([(noise.update(replace_state=True), noise().clone())[1] for _ in range(40)])
    spun_up = collect(noise, 60)[20:]
    assert first.std().item() == pytest.approx(spun_up.std().item(), rel=0.1)


@pytest.mark.parametrize("lambd", [0.25, 1.0, 2.0])
def test_lag1_autocorrelation_matches_phi(transform, lambd) -> None:
    noise = make_noise(transform, {"type": "diffusion", "sigma": 1.0, "lambd": lambd})
    fields = collect(noise, 80).flatten(start_dim=1)

    previous, current = fields[:-1], fields[1:]
    previous = previous - previous.mean(dim=1, keepdim=True)
    current = current - current.mean(dim=1, keepdim=True)
    corr = (previous * current).mean(dim=1) / (previous.std(dim=1) * current.std(dim=1))

    assert corr.mean().item() == pytest.approx(math.exp(-lambd), abs=0.05)


def test_power_spectrum_follows_the_configured_kt() -> None:
    """Measured angular power must follow the analytic (2l+1) * sigma_l^2.

    That analytic shape is also what the band filter of the conditioned noise is built from.
    """
    kT = 0.01
    # A Gaussian grid of nlat rings only integrates exactly up to l = nlat/2 - 1, so
    # generate and measure inside that band or quadrature error masks the comparison.
    band = NLAT // 2
    noise = make_noise(build_inverse_sht(NLAT, lmax=band), {"type": "diffusion", "sigma": 1.0, "kT": kT}, batch_size=8)
    fields = collect(noise, 40)

    sht = RegularSHT(nlat=NLAT, truncation=band - 1)
    # the (b, t, e, point, var) layout the transform expects
    coeffs = sht(fields.reshape(-1, 1, 1, fields.shape[-1], 1))
    measured = sht.power_spectral_density(coeffs).mean(dim=(0, 1, 2, -1))

    analytic = diffusion_degree_variance(torch.tensor([kT]), band)[0]
    # l=0 holds a single m, so it is not comparable with the rest of the spectrum.
    ratio = measured[1:].double() / analytic[1:]

    assert (ratio.std() / ratio.mean()).item() < 0.05


def test_antithetic_pairs_are_exact_negations(transform) -> None:
    noise = make_noise(transform, {"type": "diffusion", "sigma": 1.0}, centered=True)
    noise.update(replace_state=True)
    field = noise()

    torch.testing.assert_close(field[:, 0], -field[:, 1])
    torch.testing.assert_close(field[:, 2], -field[:, 3])
    # ...while distinct pairs stay independent
    assert not torch.allclose(field[:, 0].abs(), field[:, 2].abs())


def test_members_are_independent_without_centering(transform) -> None:
    noise = make_noise(transform, {"type": "white", "sigma": 1.0})
    fields = collect(noise, 30)
    members = fields.transpose(0, 2).flatten(start_dim=1)[:ENSEMBLE]
    corr = torch.corrcoef(members)

    off_diagonal = corr[~torch.eye(ENSEMBLE, dtype=torch.bool)]
    assert off_diagonal.abs().max().item() < 0.1


def test_realisation_is_keyed_to_the_global_member_index(transform) -> None:
    """Splitting members across ranks must not change any member's realisation."""
    params = {"type": "diffusion", "sigma": 1.0}
    whole = make_noise(transform, params, ensemble_size=4, num_members_total=4)
    whole.update(replace_state=True)
    reference = whole()

    halves = []
    for offset in (0, 2):
        part = make_noise(transform, params, ensemble_size=2, member_offset=offset, num_members_total=4)
        part.update(replace_state=True)
        halves.append(part())

    torch.testing.assert_close(reference, torch.cat(halves, dim=1))


def test_centering_pairs_by_global_index(transform) -> None:
    """Antithetic partners must stay paired even when they land on different ranks."""
    seeds, reflects = noise_seeds_reflects(2, centered=True, member_offset=2, num_members_total=4)
    assert seeds[0] == seeds[1]
    assert reflects == [True, False]


def test_same_seed_reproduces_the_field(transform) -> None:
    params = {"type": "diffusion", "sigma": 1.0}
    first = make_noise(transform, params)
    second = make_noise(transform, params)
    first.update(replace_state=True)
    second.update(replace_state=True)

    torch.testing.assert_close(first(), second())


def test_group_id_decorrelates_concurrent_batches(transform) -> None:
    params = {"type": "diffusion", "sigma": 1.0}
    first = make_noise(transform, params, group_id=0)
    second = make_noise(transform, params, group_id=1)
    first.update(replace_state=True)
    second.update(replace_state=True)

    assert not torch.allclose(first(), second())


def test_history_is_temporally_correlated(transform) -> None:
    """A resampled multi-step history must carry the OU correlation, not independent steps."""
    lambd = 1.0
    noise = make_noise(transform, {"type": "diffusion", "sigma": 1.0, "lambd": lambd}, num_time_steps=2, batch_size=8)
    noise.update(replace_state=True)
    field = noise()

    assert field.shape[2] == 2
    earlier, later = field[:, :, 0].flatten(), field[:, :, 1].flatten()
    corr = ((earlier - earlier.mean()) * (later - later.mean())).mean() / (earlier.std() * later.std())

    assert corr.item() == pytest.approx(math.exp(-lambd), abs=0.05)
    assert earlier.std().item() == pytest.approx(later.std().item(), rel=0.1)


def test_per_channel_kt_and_lambd(transform) -> None:
    seeds, reflects = noise_seeds_reflects(ENSEMBLE, centered=False)
    noise = build_noise(
        {"type": "diffusion", "sigma": 1.0, "kT": [1e-4, 1e-1], "lambd": [0.5, 2.0]},
        transform=transform,
        batch_size=2,
        ensemble_size=ENSEMBLE,
        num_channels=2,
        num_time_steps=1,
        seeds=seeds,
        reflects=reflects,
    )
    noise.update(replace_state=True)
    field = noise()

    # The smaller kT keeps more high-degree power, so it is the rougher field.
    roughness = [field[:, :, :, channel].diff(dim=-1).std().item() for channel in (0, 1)]
    assert roughness[0] > roughness[1]


def test_per_channel_length_is_validated(transform) -> None:
    with pytest.raises(ValueError, match="entries"):
        make_noise(transform, {"type": "diffusion", "kT": [1e-4, 1e-1, 1e-2]})


def test_output_shape(transform) -> None:
    _, _, num_grid_points = transform
    noise = make_noise(transform, {"type": "diffusion"}, batch_size=3, num_time_steps=2)
    noise.update(replace_state=True)
    assert noise().shape == (3, ENSEMBLE, 2, 1, num_grid_points)


def test_batch_size_is_resized_on_demand(transform) -> None:
    noise = make_noise(transform, {"type": "diffusion"}, batch_size=1)
    noise.update(replace_state=True, batch_size=5)
    assert noise().shape[0] == 5


def test_dummy_noise_is_zero_and_skips_the_transform(transform) -> None:
    _, _, num_grid_points = transform
    noise = make_noise(transform, {"type": "dummy", "mode": "constant_zero"})
    noise.update(replace_state=True)

    assert isinstance(noise, DummyNoiseS2)
    assert not noise.is_stateful()
    assert noise().shape[-1] == num_grid_points
    assert torch.count_nonzero(noise()) == 0


def test_set_tensor_state_rejects_a_mismatched_layout(transform) -> None:
    noise = make_noise(transform, {"type": "diffusion"})
    with pytest.raises(ValueError, match="shape mismatch beyond batch dim"):
        noise.set_tensor_state(torch.zeros(2, ENSEMBLE, 1, 1, 4, 4, 2))


def test_state_is_not_checkpointed(transform) -> None:
    """The state is per-run scratch; restoring a checkpoint must not revive a stale field."""
    noise = make_noise(transform, {"type": "diffusion"})
    assert "state" not in noise.state_dict()
    assert "sigma_l" not in noise.state_dict()


# The 100/200/400/800 km FourCastNet 3 rungs.
FCN3_KT = torch.tensor([1.2322e-4, 4.9289e-4, 1.9716e-3, 7.8862e-3])


def test_band_limits_of_the_fcn3_ladder_at_n320() -> None:
    """Each rung holds 99% of its variance below roughly 6370 km / L * 3, halving per rung."""
    assert diffusion_band_limit(FCN3_KT, 640).tolist() == [193, 96, 48, 24]


def test_heat_kernel_keeps_the_mean_and_smooths_more_for_larger_scales() -> None:
    response = heat_kernel_response(FCN3_KT, 64)

    assert torch.all(response[:, 0] == 1.0)
    assert torch.all(response.diff(dim=-1) <= 0)
    assert torch.all(response[1:, 1:] < response[:-1, 1:])


def test_lowpass_passes_the_band_and_rolls_off_beyond_it() -> None:
    lmax = 640
    response = diffusion_lowpass_response(FCN3_KT, lmax, quantile=0.99, taper=0.25)
    limits = diffusion_band_limit(FCN3_KT, lmax, 0.99)

    for channel, limit in enumerate(limits.tolist()):
        end = math.ceil(limit * 1.25)
        assert torch.all(response[channel, : limit + 1] == 1.0)
        assert torch.all(response[channel, end:] == 0.0)
        roll_off = response[channel, limit : end + 1]
        assert torch.all(roll_off.diff() < 0)


def test_lowpass_without_taper_is_a_hard_cut() -> None:
    response = diffusion_lowpass_response(FCN3_KT[:1], 640, quantile=0.99, taper=0.0)
    assert response[0].sum().item() == 194
    assert set(response.unique().tolist()) == {0.0, 1.0}


def test_full_quantile_passes_every_degree() -> None:
    assert torch.all(diffusion_lowpass_response(torch.tensor([1e-6]), 16, quantile=1.0, taper=0.0) == 1.0)


@pytest.mark.parametrize("quantile, taper", [(0.0, 0.25), (1.5, 0.25), (0.99, -0.1)])
def test_lowpass_rejects_invalid_parameters(quantile, taper) -> None:
    with pytest.raises(ValueError):
        diffusion_lowpass_response(FCN3_KT, 64, quantile=quantile, taper=taper)


def test_filter_truncation_covers_every_significant_degree() -> None:
    lmax = 640
    responses = [diffusion_lowpass_response(FCN3_KT, lmax), heat_kernel_response(FCN3_KT, lmax)]
    truncation = filter_truncation(responses, lmax)

    # the 100 km rung's roll-off ends at ceil(193 * 1.25) = 242, well inside the 319 degrees
    # an N320 grid analyses exactly
    assert truncation == 241
    assert filter_truncation([torch.ones(2, 16)], 16) == 15


def test_per_channel_values() -> None:
    torch.testing.assert_close(per_channel_values(0.5, 3, "kT"), torch.full((3,), 0.5))
    torch.testing.assert_close(per_channel_values([1, 2], 2, "kT"), torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError, match="entries"):
        per_channel_values([1, 2], 3, "kT")


# --- per-channel spectra ------------------------------------------------------------------


def make_channels(transform, coefficient_variance, *, batch_size=2, ensemble_size=ENSEMBLE, num_time_steps=1):
    seeds, reflects = noise_seeds_reflects(ensemble_size, centered=False)
    return build_noise(
        {"type": "diffusion", "sigma": 1.0, "lambd": 1.0},
        transform=transform,
        batch_size=batch_size,
        ensemble_size=ensemble_size,
        num_channels=coefficient_variance.shape[0],
        num_time_steps=num_time_steps,
        seeds=seeds,
        reflects=reflects,
        coefficient_variance=coefficient_variance,
    )


# A spread-like spectrum: rising to degree ~6, falling steeply beyond.
TABLE = {"degree": [1.0, 3.0, 6.0, 12.0, 24.0], "sigma2": [1.0, 0.8, 0.5, 0.05, 1e-6]}


def test_tabulated_spectrum_interpolates_in_log_log_space() -> None:
    values = tabulated_coefficient_variance(TABLE["degree"], TABLE["sigma2"], 32)

    assert values.dtype == torch.float64
    assert values[0] == 0.0  # the global mean is not a spatial scale
    for degree, sigma2 in zip(TABLE["degree"], TABLE["sigma2"]):
        assert values[int(degree)].item() == pytest.approx(sigma2, rel=1e-12)
    # between entries, log(sigma2) is linear in log(l)
    weight = math.log(4 / 3) / math.log(6 / 3)
    assert values[4].item() == pytest.approx(math.exp((1 - weight) * math.log(0.8) + weight * math.log(0.5)), rel=1e-12)
    assert torch.all(values[24:] == values[24])  # held flat beyond the table


@pytest.mark.parametrize(
    "degree, sigma2",
    [([1.0], [1.0]), ([1.0, 2.0], [1.0]), ([2.0, 1.0], [1.0, 1.0]), ([0.5, 2.0], [1.0, 1.0]), ([1.0, 2.0], [1.0, 0.0])],
)
def test_tabulated_spectrum_rejects_invalid_tables(degree, sigma2) -> None:
    with pytest.raises(ValueError):
        tabulated_coefficient_variance(degree, sigma2, 16)


def test_band_limit_of_a_kt_channel_is_unchanged() -> None:
    kT = torch.tensor([1.2322e-4, 7.8862e-3])
    variance = degree_variance(heat_kernel_response(kT, 640))

    torch.testing.assert_close(band_limit(variance), diffusion_band_limit(kT, 640))
    torch.testing.assert_close(lowpass_response(variance), diffusion_lowpass_response(kT, 640))


def test_lowpass_follows_a_tabulated_spectrum() -> None:
    variance = degree_variance(tabulated_coefficient_variance(TABLE["degree"], TABLE["sigma2"], 64))[None]
    response = lowpass_response(variance, quantile=0.99, taper=0.25)
    limit = int(band_limit(variance, 0.99)[0])

    assert 6 < limit < 30
    assert torch.all(response[0, : limit + 1] == 1.0)
    assert torch.all(response[0, math.ceil(limit * 1.25) :] == 0.0)


def test_kt_channel_through_its_coefficient_variance_matches_the_kt_path(transform) -> None:
    kT = 0.01
    reference = make_noise(transform, {"type": "diffusion", "sigma": 1.0, "lambd": 1.0, "kT": kT})
    explicit = make_channels(transform, heat_kernel_response(torch.tensor([kT]), NLAT))
    for noise in (reference, explicit):
        noise.update(replace_state=True)

    torch.testing.assert_close(explicit(), reference(), atol=1e-5, rtol=1e-4)


def test_tabulated_channel_has_the_stationary_variance(transform) -> None:
    """Area-weighted, and with power at small scales: a large-scale field has too few modes to pin it down."""
    table = tabulated_coefficient_variance([1.0, 8.0, 16.0, 31.0], [0.01, 1.0, 0.5, 0.01], NLAT)
    noise = make_channels(transform, table[None], batch_size=4)
    fields = collect(noise, 40)
    area = torch.as_tensor(quadrature_weights([2 * NLAT] * NLAT)).float()

    assert torch.sqrt((fields**2 * area).sum(dim=-1).mean()).item() == pytest.approx(1.0, rel=0.03)


def test_tabulated_channel_power_follows_the_table() -> None:
    band = NLAT // 2
    table = tabulated_coefficient_variance(TABLE["degree"], TABLE["sigma2"], band)
    noise = make_channels(build_inverse_sht(NLAT, lmax=band), table[None], batch_size=8)
    fields = collect(noise, 40)

    sht = RegularSHT(nlat=NLAT, truncation=band - 1)
    coeffs = sht(fields.reshape(-1, 1, 1, fields.shape[-1], 1))
    measured = sht.power_spectral_density(coeffs).mean(dim=(0, 1, 2, -1))
    ratio = measured[1:].double() / degree_variance(table)[1:]

    assert (ratio.std() / ratio.mean()).item() < 0.05


def test_coefficient_variance_is_validated(transform) -> None:
    with pytest.raises(ValueError, match="expected"):
        make_channels(transform, torch.ones(2, NLAT + 1))
    with pytest.raises(ValueError, match="non-negative"):
        make_channels(transform, -torch.ones(1, NLAT))
    with pytest.raises(ValueError, match="diffusion"):
        seeds, reflects = noise_seeds_reflects(1, centered=False)
        build_noise(
            {"type": "white"},
            transform=transform,
            batch_size=1,
            ensemble_size=1,
            num_channels=1,
            num_time_steps=1,
            seeds=seeds,
            reflects=reflects,
            coefficient_variance=torch.ones(1, NLAT),
        )


# --- in-place updates and chunked transforms ----------------------------------------------


def _known_draws(noise, seed: int = 7):
    """Make the noise draw reproducible standard normals we can also compute with."""
    generator = torch.Generator().manual_seed(seed)
    draws = []

    def draw(out):
        values = torch.randn(out.shape, generator=generator)
        draws.append(values.clone())
        return out.copy_(values)

    noise._draw_normal = draw
    return draws


@pytest.mark.parametrize("num_time_steps", [1, 2, 4])
def test_redrawn_history_is_the_toeplitz_discount_of_the_draws(transform, num_time_steps) -> None:
    lambd = [0.5, 1.5]
    seeds, reflects = noise_seeds_reflects(2, centered=False)
    noise = build_noise(
        {"type": "diffusion", "sigma": 1.0, "kT": [1e-3, 1e-2], "lambd": lambd},
        transform=transform,
        batch_size=1,
        ensemble_size=2,
        num_channels=2,
        num_time_steps=num_time_steps,
        seeds=seeds,
        reflects=reflects,
    )
    draws = _known_draws(noise)
    noise.update(replace_state=True)

    scaled = noise.sigma_l * draws[0]
    phi = torch.exp(-torch.tensor(lambd))
    scaled[:, :, 0] = scaled[:, :, 0] / torch.sqrt(1 - phi**2)[None, None, :, None, None, None]
    # discount[c, t, r] = phi_c^(t - r) for r <= t
    steps = torch.arange(num_time_steps)
    lags = (steps[:, None] - steps[None, :]).clamp(min=0)
    discount = torch.where(steps[:, None] >= steps[None, :], phi[:, None, None] ** lags, 0.0)
    expected = torch.einsum("ctr,berclmu->betclmu", discount, scaled)

    torch.testing.assert_close(noise.state, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("num_time_steps", [1, 3])
def test_rollout_step_slides_the_history_and_advances_the_newest(transform, num_time_steps) -> None:
    noise = make_noise(transform, {"type": "diffusion", "sigma": 1.0, "lambd": 1.0}, num_time_steps=num_time_steps)
    noise.update(replace_state=True)
    previous = noise.state.clone()
    draws = _known_draws(noise)

    noise.update()

    expected_newest = math.exp(-1.0) * previous[:, :, -1] + (noise.sigma_l * draws[0])[:, :, 0]
    torch.testing.assert_close(noise.state[:, :, -1], expected_newest, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(noise.state[:, :, :-1], previous[:, :, 1:])


def test_chunked_transform_matches_the_whole(transform, monkeypatch) -> None:
    seeds, reflects = noise_seeds_reflects(ENSEMBLE, centered=False)
    noise = build_noise(
        {"type": "diffusion", "sigma": 1.0, "kT": [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]},
        transform=transform,
        batch_size=2,
        ensemble_size=ENSEMBLE,
        num_channels=5,
        num_time_steps=2,
        seeds=seeds,
        reflects=reflects,
    )
    noise.update(replace_state=True)
    monkeypatch.setattr(DiffusionNoiseS2, "transform_chunk_channels", None)
    whole = noise()
    monkeypatch.setattr(DiffusionNoiseS2, "transform_chunk_channels", 2)

    torch.testing.assert_close(noise(), whole)
    torch.testing.assert_close(noise.transform_channels(slice(1, 3)), whole[:, :, :, 1:3])
