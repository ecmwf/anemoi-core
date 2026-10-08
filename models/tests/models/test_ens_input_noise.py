# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from anemoi.models.layers.ensemble import SphericalInputConditionedNoise
from anemoi.models.layers.ensemble import SphericalInputNoise
from anemoi.models.layers.ensemble import kT_from_length_scale
from anemoi.models.layers.spectral_helpers import quadrature_weights
from anemoi.models.layers.spectral_transforms import SphericalSpectralFilter
from anemoi.models.layers.spherical_noise import band_limit
from anemoi.models.layers.spherical_noise import degree_variance
from anemoi.models.layers.spherical_noise import heat_kernel_response
from anemoi.models.models.ens_encoder_processor_decoder import AnemoiEnsModelEncProcDec

NLAT = 16
NUM_POINTS = NLAT * 2 * NLAT
NUM_VARS = 5
NUM_NOISE = 3
N_STEP_INPUT = 2


@pytest.fixture
def input_noise() -> SphericalInputNoise:
    return SphericalInputNoise(
        grid=NLAT,
        noise={"type": "diffusion", "sigma": 1.0, "lambd": 1.0},
        n_channels=NUM_NOISE,
        dataset="era5",
        num_time_steps=N_STEP_INPUT,
        num_grid_points=NUM_POINTS,
    )


class _NodeAttributes(nn.Module):
    attr_ndims = {"era5": 4}
    num_nodes = {"era5": NUM_POINTS}

    def forward(self, name: str, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size * NUM_POINTS, self.attr_ndims[name])


class _NoResidual(nn.Module):
    def forward(self, x, **kwargs):
        return None


class _StubEnsModel(AnemoiEnsModelEncProcDec):
    """Exercises the input-noise wiring without standing up a whole graph model."""

    def __init__(self, input_noise: SphericalInputNoise | None) -> None:
        nn.Module.__init__(self)
        self.n_step_input = N_STEP_INPUT
        self.n_step_output = 1
        self.condition_on_residual = False
        self.num_input_channels = {"era5": NUM_VARS}
        self.input_noise = input_noise
        self.input_noise_dataset = "era5" if input_noise is not None else None
        self._encoder_input_keep_idx = None
        self.node_attributes = _NodeAttributes()
        self.residual = nn.ModuleDict({"era5": _NoResidual()})


def test_input_dim_grows_by_channels_times_time_steps(input_noise) -> None:
    without = _StubEnsModel(None)._calculate_input_dim("era5")
    with_noise = _StubEnsModel(input_noise)._calculate_input_dim("era5")

    assert with_noise - without == N_STEP_INPUT * NUM_NOISE


def test_noise_is_interleaved_per_time_step(input_noise) -> None:
    """Noise must sit alongside the data of its own time step, as makani's flatten_history does."""
    batch, ensemble = 2, 3
    x = torch.zeros(batch, N_STEP_INPUT, ensemble, NUM_POINTS, NUM_VARS)
    noise = torch.ones(batch, ensemble, N_STEP_INPUT, NUM_NOISE, NUM_POINTS)
    for time in range(N_STEP_INPUT):
        noise[:, :, time] *= time + 1

    model = _StubEnsModel(input_noise)
    x_data_latent, _, _ = model._assemble_input(
        x,
        fcstep=0,
        batch_ens_size=batch * ensemble,
        dataset_name="era5",
        input_noise=noise,
    )

    features = x_data_latent[:, : N_STEP_INPUT * (NUM_VARS + NUM_NOISE)]
    per_step = features.reshape(-1, N_STEP_INPUT, NUM_VARS + NUM_NOISE)
    for time in range(N_STEP_INPUT):
        assert torch.all(per_step[:, time, :NUM_VARS] == 0)
        assert torch.all(per_step[:, time, NUM_VARS:] == time + 1)


def test_assemble_input_is_unchanged_without_noise() -> None:
    batch, ensemble = 2, 3
    x = torch.randn(batch, N_STEP_INPUT, ensemble, NUM_POINTS, NUM_VARS)
    model = _StubEnsModel(None)

    x_data_latent, _, _ = model._assemble_input(
        x, fcstep=0, batch_ens_size=batch * ensemble, dataset_name="era5", input_noise=None
    )

    assert x_data_latent.shape[-1] == N_STEP_INPUT * NUM_VARS + 4 + 1


def test_grid_mismatch_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="would not align with the data nodes"):
        SphericalInputNoise(
            grid=NLAT,
            noise={"type": "diffusion"},
            n_channels=1,
            num_time_steps=1,
            num_grid_points=NUM_POINTS + 1,
        )


def test_field_is_built_lazily(input_noise) -> None:
    assert input_noise.noise is None
    with pytest.raises(RuntimeError, match="called before advance"):
        input_noise.sample()

    input_noise.advance(fcstep=0, batch_size=2, ensemble_size=4)
    assert input_noise.sample().shape == (2, 4, N_STEP_INPUT, NUM_NOISE, NUM_POINTS)


def test_rollout_replaces_the_state_only_at_the_first_step(input_noise) -> None:
    calls = []
    input_noise.advance(fcstep=0, batch_size=1, ensemble_size=2)
    original = input_noise.noise.update
    input_noise.noise.update = lambda **kwargs: (calls.append(kwargs["replace_state"]), original(**kwargs))[1]

    for fcstep in range(1, 4):
        input_noise.advance(fcstep=fcstep, batch_size=1, ensemble_size=2)
    input_noise.advance(fcstep=0, batch_size=1, ensemble_size=2)

    assert calls == [False, False, False, True]


def test_rollout_steps_stay_correlated(input_noise) -> None:
    """An autoregressive step must evolve the perturbation, not redraw it."""
    input_noise.advance(fcstep=0, batch_size=4, ensemble_size=2)
    first = input_noise.sample().flatten()
    input_noise.advance(fcstep=1, batch_size=4, ensemble_size=2)
    second = input_noise.sample().flatten()

    corr = ((first - first.mean()) * (second - second.mean())).mean() / (first.std() * second.std())
    assert 0.2 < corr.item() < 0.6  # phi = exp(-1) = 0.368


def test_noise_history_slides_in_lockstep_with_the_data(input_noise) -> None:
    """The noise history must shift by exactly one step per rollout step.

    `Forecaster.advance_input` rolls the data history by one, so the noise history has
    to shift identically or the perturbation on the older input step would no longer
    correspond to the field that step was predicted under.
    """
    input_noise.advance(fcstep=0, batch_size=2, ensemble_size=2)
    previous = input_noise.sample().clone()

    for fcstep in range(1, 4):
        input_noise.advance(fcstep=fcstep, batch_size=2, ensemble_size=2)
        current = input_noise.sample()
        # time slot 0 of this step is time slot 1 of the previous step
        torch.testing.assert_close(current[:, :, 0], previous[:, :, 1])
        assert not torch.allclose(current[:, :, 1], previous[:, :, 1])
        previous = current.clone()


def test_rollout_restart_breaks_the_history(input_noise) -> None:
    """A new rollout must not inherit the previous one's trajectory."""
    input_noise.advance(fcstep=0, batch_size=2, ensemble_size=2)
    input_noise.advance(fcstep=1, batch_size=2, ensemble_size=2)
    last_of_rollout = input_noise.sample().clone()

    input_noise.advance(fcstep=0, batch_size=2, ensemble_size=2)
    first_of_next = input_noise.sample()

    assert not torch.allclose(first_of_next[:, :, 0], last_of_rollout[:, :, 1])


def test_field_is_rebuilt_when_the_ensemble_layout_changes(input_noise) -> None:
    input_noise.advance(fcstep=0, batch_size=1, ensemble_size=2, member_offset=0, num_members_total=4)
    first = input_noise.noise

    input_noise.advance(fcstep=0, batch_size=1, ensemble_size=2, member_offset=2, num_members_total=4)
    assert input_noise.noise is not first

    input_noise.advance(fcstep=1, batch_size=1, ensemble_size=2, member_offset=2, num_members_total=4)
    assert input_noise.noise is not None


def test_accepts_an_omegaconf_config() -> None:
    """Hydra passes ListConfig/DictConfig, not plain Python containers."""
    config = OmegaConf.create(
        {
            "_target_": "anemoi.models.layers.ensemble.SphericalInputNoise",
            "grid": NLAT,
            "n_channels": 2,
            "noise": {"type": "diffusion", "sigma": 1.0, "kT": [1.0e-4, 1.0e-1], "lambd": [0.5, 2.0]},
        }
    )
    noise = instantiate(config, _recursive_=False, num_time_steps=1, num_grid_points=NUM_POINTS)

    noise.advance(fcstep=0, batch_size=1, ensemble_size=2)
    assert noise.sample().shape == (1, 2, 1, 2, NUM_POINTS)


# --- named channels and seeds -------------------------------------------------------------


def test_named_kt_channels_match_the_fcn3_ladder() -> None:
    kT = [0.05, 0.1]
    ladder = SphericalInputNoise(
        grid=NLAT, noise={"type": "diffusion", "kT": kT}, n_channels=2, num_time_steps=N_STEP_INPUT
    )
    named = SphericalInputNoise(
        grid=NLAT,
        noise={"type": "diffusion"},
        channels={"small": {"kT": kT[0]}, "large": {"kT": kT[1]}},
        num_time_steps=N_STEP_INPUT,
    )
    assert named.n_channels == 2

    for fcstep in range(2):
        for noise in (ladder, named):
            noise.advance(fcstep=fcstep, batch_size=2, ensemble_size=2, seed=1)
        torch.testing.assert_close(named.sample(), ladder.sample(), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    "noise, channels, match",
    [
        ({"type": "diffusion"}, {"a": {"kT": 0.1, "spread": ["t"]}}, "SphericalInputConditionedNoise"),
        ({"type": "diffusion", "kT": [0.1]}, {"a": {"kT": 0.1}}, "per channel"),
        ({"type": "white"}, {"a": {"kT": 0.1}}, "diffusion"),
        ({"type": "diffusion"}, {"a": {"kT": 0.1, "spectrum": {"degree": [1, 2], "sigma2": [1, 1]}}}, "exactly one"),
        ({"type": "diffusion"}, {"a": {"kT": 0.1, "spred": ["t"]}}, "unknown keys"),
    ],
)
def test_invalid_channels_are_rejected(noise, channels, match) -> None:
    with pytest.raises(ValueError, match=match):
        SphericalInputNoise(grid=NLAT, noise=noise, channels=channels, num_time_steps=1)


def test_channel_count_must_agree_with_n_channels() -> None:
    with pytest.raises(ValueError, match="n_channels=3"):
        SphericalInputNoise(grid=NLAT, noise={"type": "diffusion"}, channels={"a": {"kT": 0.1}}, n_channels=3)


def test_seed_sets_the_noise_stream(input_noise) -> None:
    other = SphericalInputNoise(
        grid=NLAT,
        noise={"type": "diffusion", "sigma": 1.0, "lambd": 1.0},
        n_channels=NUM_NOISE,
        num_time_steps=N_STEP_INPUT,
    )
    input_noise.advance(fcstep=0, batch_size=1, ensemble_size=2, seed=11)
    other.advance(fcstep=0, batch_size=1, ensemble_size=2, seed=11)
    torch.testing.assert_close(input_noise.sample(), other.sample())

    other.advance(fcstep=0, batch_size=1, ensemble_size=2, seed=12)
    assert not torch.allclose(input_noise.sample(), other.sample())


def test_without_a_seed_the_noise_follows_torch() -> None:
    """How inference draws different members: each run is seeded differently."""

    def draw(torch_seed: int) -> torch.Tensor:
        torch.manual_seed(torch_seed)
        noise = SphericalInputNoise(grid=NLAT, noise={"type": "diffusion"}, n_channels=1, num_time_steps=1)
        noise.advance(fcstep=0, batch_size=1, ensemble_size=1)
        return noise.sample()

    torch.testing.assert_close(draw(3), draw(3))
    assert not torch.allclose(draw(3), draw(4))


# --- spread-conditioned noise -------------------------------------------------------------

# Model input of the stub: two plain variables interleaved with three spread fields. The stub's
# data and model inputs coincide, so the statistics are indexed the same way.
STD_NAME_TO_INDEX = {"a": 0, "std_t_850": 1, "b": 2, "std_q_850": 3, "std_2t": 4}
STD_IDX = [1, 3, 4]
KEPT_IDX = [0, 2]
STATISTICS = {
    "mean": np.array([0.0, 2.0, 0.0, 0.5, 1.0]),
    "stdev": np.array([1.0, 1.0, 1.0, 0.25, 0.5]),
    "maximum": np.array([1.0, 8.0, 1.0, 2.0, 4.0]),
}
SPREAD_MEAN = torch.tensor([2.0, 0.5, 1.0])  # of the spread inputs, in input order
# Every band ends well below the grid's degree 15: kT large, and a table falling off by degree 8.
SPECTRUM = {"degree": [1.0, 2.0, 4.0, 8.0], "sigma2": [1.0, 0.6, 0.1, 1e-6]}
CHANNELS = {
    "t": {"kT": 0.05, "spread": ["t_850", "2t"]},
    "q": {"spectrum": SPECTRUM, "spread": ["q_850"]},
    "large": {"kT": 0.1},
}
SCALED, UNSCALED = [0, 1], [2]
MODULATION = {"reference": "climatology", "normalizer": "none", "smoothing_km": None}


def noise_params() -> dict:
    return {"type": "diffusion", "sigma": 1.0, "lambd": 1.0}


def unscaled_channels() -> dict:
    return {name: {key: value for key, value in spec.items() if key != "spread"} for name, spec in CHANNELS.items()}


def make_plain(**kwargs) -> SphericalInputNoise:
    params = {"grid": NLAT, "noise": noise_params(), "channels": unscaled_channels(), "dataset": "era5"}
    params.update(num_time_steps=N_STEP_INPUT, num_grid_points=NUM_POINTS, **kwargs)
    return SphericalInputNoise(**params)


def make_conditioned(modulation: dict | None = None, channels: dict | None = None, **kwargs):
    params = {"grid": NLAT, "noise": noise_params(), "channels": channels or CHANNELS, "dataset": "era5"}
    params.update(num_time_steps=N_STEP_INPUT, num_grid_points=NUM_POINTS, **kwargs)
    return SphericalInputConditionedNoise(
        modulation={**MODULATION, **(modulation or {})},
        name_to_index=STD_NAME_TO_INDEX,
        statistics=STATISTICS,
        name_to_index_stats=STD_NAME_TO_INDEX,
        **params,
    )


def spread_inputs(batch: int = 2, seed: int = 0, constant: bool = False) -> torch.Tensor:
    """Positive spread fields around their climatological mean, shape (batch, time, points, variables)."""
    if constant:
        return SPREAD_MEAN.expand(batch, N_STEP_INPUT, NUM_POINTS, len(STD_IDX)).clone()
    generator = torch.Generator().manual_seed(seed)
    ratio = torch.rand(batch, N_STEP_INPUT, NUM_POINTS, len(STD_IDX), generator=generator) * 3.0 + 0.1
    return ratio * SPREAD_MEAN


def advance(noise, fcstep: int, inputs: torch.Tensor | None = None, ensemble_size: int = 2) -> torch.Tensor:
    batch = inputs.shape[0] if inputs is not None else 2
    noise.advance(fcstep=fcstep, batch_size=batch, ensemble_size=ensemble_size, inputs=inputs, seed=5)
    return noise.sample().clone()


def area_mean(field: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...p,p->...", field, torch.as_tensor(quadrature_weights([2 * NLAT] * NLAT)).float())


class _ConfiguredStubEnsModel(_StubEnsModel):
    """Builds its input noise through the model's own `_build_input_noise`."""

    def __init__(self, config: dict) -> None:
        super().__init__(None)
        self._input_noise_config = OmegaConf.create(config)
        self.input_datasets = ["era5"]
        self.statistics = {"era5": STATISTICS}
        indices = SimpleNamespace(input=SimpleNamespace(name_to_index=STD_NAME_TO_INDEX))
        self.data_indices = {"era5": SimpleNamespace(model=indices, data=indices)}
        self._build_input_noise()


def conditioned_config(**modulation) -> dict:
    return {
        "_target_": "anemoi.models.layers.ensemble.SphericalInputConditionedNoise",
        "grid": NLAT,
        "noise": noise_params(),
        "channels": CHANNELS,
        "modulation": {**MODULATION, **modulation},
    }


def test_conditioned_noise_replaces_its_inputs_in_the_encoder_width() -> None:
    model = _ConfiguredStubEnsModel(conditioned_config())
    without = _StubEnsModel(None)._calculate_input_dim("era5")

    assert model.input_noise.consumed_input_idx == STD_IDX
    assert model.input_noise.modulated_channels == SCALED
    assert model._calculate_input_dim("era5") - without == N_STEP_INPUT * (NUM_NOISE - len(STD_IDX))


def test_disabled_modulation_still_drops_the_spread_inputs() -> None:
    """The matched baseline must see exactly the same encoder inputs as the experiment."""
    model = _ConfiguredStubEnsModel(conditioned_config(enabled=False))

    assert not model.input_noise.conditioned
    assert model._encoder_input_keep_idx.tolist() == KEPT_IDX


def test_spread_inputs_are_dropped_before_the_noise_is_appended() -> None:
    model = _ConfiguredStubEnsModel(conditioned_config())
    batch, ensemble = 2, 3
    x = torch.arange(1, NUM_VARS + 1, dtype=torch.float32).expand(batch, N_STEP_INPUT, ensemble, NUM_POINTS, NUM_VARS)
    noise = torch.full((batch, ensemble, N_STEP_INPUT, NUM_NOISE, NUM_POINTS), -1.0)

    x_data_latent, _, _ = model._assemble_input(
        x, fcstep=0, batch_ens_size=batch * ensemble, dataset_name="era5", input_noise=noise
    )

    width = len(KEPT_IDX) + NUM_NOISE
    per_step = x_data_latent[:, : N_STEP_INPUT * width].reshape(-1, N_STEP_INPUT, width)
    assert torch.all(per_step[..., : len(KEPT_IDX)] == torch.tensor([1.0, 3.0]))
    assert torch.all(per_step[..., len(KEPT_IDX) :] == -1.0)
    assert x_data_latent.shape[-1] == model._calculate_input_dim("era5")


def test_noise_inputs_are_the_spread_channels_of_the_first_member() -> None:
    model = _ConfiguredStubEnsModel(conditioned_config())
    x = torch.randn(2, N_STEP_INPUT, 3, NUM_POINTS, NUM_VARS)

    torch.testing.assert_close(model._input_noise_inputs(x, None, None), x[:, :, 0][..., STD_IDX])


class _NoiseAdvanced(RuntimeError):
    pass


class _ForwardStubEnsModel(_ConfiguredStubEnsModel):
    """Runs the real `forward` up to the point the noise has been advanced and sampled."""

    _graph_name_hidden = "hidden"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.advance_calls = []
        original = self.input_noise.advance

        def spy(**kwargs):
            self.advance_calls.append(kwargs)
            return original(**kwargs)

        self.input_noise.advance = spy

        def stop(name, batch_size):
            raise _NoiseAdvanced

        self.node_attributes.forward = stop


@pytest.mark.parametrize("fcstep, expects_inputs", [(0, True), (1, False), (3, False)])
def test_forward_passes_the_spread_only_at_the_first_step(fcstep, expects_inputs) -> None:
    model = _ForwardStubEnsModel(conditioned_config())
    x = {"era5": (torch.rand(1, N_STEP_INPUT, 2, NUM_POINTS, NUM_VARS) + 0.5) * 2.0}
    if fcstep > 0:
        with pytest.raises(_NoiseAdvanced):
            model.forward(x, fcstep=0, input_noise_seed=17)

    with pytest.raises(_NoiseAdvanced):
        model.forward(x, fcstep=fcstep, input_noise_seed=17)

    call = model.advance_calls[-1]
    assert call["seed"] == 17
    if expects_inputs:
        torch.testing.assert_close(call["inputs"], x["era5"][:, :, 0][..., STD_IDX])
    else:
        assert call["inputs"] is None


def test_disabled_modulation_is_the_same_channels_unscaled() -> None:
    conditioned = make_conditioned({"enabled": False})
    plain = make_plain()

    for fcstep in range(3):
        torch.testing.assert_close(advance(conditioned, fcstep), advance(plain, fcstep))


def test_channels_without_spread_are_untouched() -> None:
    conditioned = make_conditioned()
    plain = make_plain()

    for fcstep, inputs in ((0, spread_inputs()), (1, None)):
        scaled, unscaled = advance(conditioned, fcstep, inputs), advance(plain, fcstep)
        torch.testing.assert_close(scaled[:, :, :, UNSCALED], unscaled[:, :, :, UNSCALED])
        assert not torch.allclose(scaled[:, :, :, SCALED], unscaled[:, :, :, SCALED])


def test_spread_at_its_climatology_with_an_all_pass_band_leaves_the_noise_unchanged() -> None:
    conditioned = make_conditioned({"rescale": "none", "band_filter": {"quantile": 1.0, "taper": 0.0}})
    plain = make_plain()

    torch.testing.assert_close(
        advance(conditioned, 0, spread_inputs(constant=True)), advance(plain, 0), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(advance(conditioned, 1), advance(plain, 1), atol=1e-5, rtol=1e-5)


def test_noise_is_stronger_where_the_spread_is_higher() -> None:
    conditioned = make_conditioned()
    inputs = spread_inputs(batch=4, constant=True)
    north = NUM_POINTS // 2  # rings run north to south
    inputs[:, :, :north] *= 3.0

    field = advance(conditioned, 0, inputs, ensemble_size=4)[:, :, :, SCALED]

    assert field[..., :north].pow(2).mean() > 3.0 * field[..., north:].pow(2).mean()


@pytest.mark.parametrize("reference, ratio", [("climatology", 4.0), ("sample_mean", 1.0)])
def test_climatology_keeps_how_uncertain_the_whole_analysis_is(reference, ratio) -> None:
    """Twice the spread everywhere: twice the noise against the climatology, unchanged per sample."""
    modulation = {"reference": reference, "clip": None, "band_filter": {"quantile": 1.0, "taper": 0.0}}
    inputs = spread_inputs()

    once = advance(make_conditioned(modulation), 0, inputs)[:, :, :, SCALED]
    twice = advance(make_conditioned(modulation), 0, 2.0 * inputs)[:, :, :, SCALED]

    torch.testing.assert_close(area_mean(twice**2), ratio * area_mean(once**2), rtol=1e-4, atol=1e-6)


def test_fixed_rescale_brings_the_long_term_mean_square_to_one() -> None:
    conditioned = make_conditioned()
    mean_square = 1.0 + (STATISTICS["stdev"][STD_IDX] / STATISTICS["mean"][STD_IDX]) ** 2
    expected = torch.tensor([(mean_square[0] + mean_square[2]) / 2, mean_square[1]], dtype=torch.float32) ** -0.5

    torch.testing.assert_close(conditioned.std_modulation.channel_scale, expected)


@pytest.mark.parametrize("smoothing_km, raw_leakage", [(None, 1e-3), (2000.0, 1e-5)])
def test_each_scaled_channel_is_confined_to_its_own_band(smoothing_km, raw_leakage) -> None:
    """Scaling leaks energy to small scales; the band filter must return each channel to its band.

    Smoothing the multiplier already removes most of the leakage, the band filter the rest.
    """
    conditioned = make_conditioned({"smoothing_km": smoothing_km})
    plain = make_plain()
    inputs = spread_inputs(batch=2)

    field = advance(conditioned, 0, inputs)
    raw_product = advance(plain, 0)[:, :, :, SCALED] * conditioned.std_modulation(inputs).unsqueeze(1)

    # the module's own filter stops at the band edge; measure at full resolution
    full_resolution = SphericalSpectralFilter([2 * NLAT] * NLAT, truncation=NLAT - 1)
    assert conditioned.spectral_filter.truncation < full_resolution.truncation
    coeffs = full_resolution.analyse(field[:, :, :, SCALED])
    raw_coeffs = full_resolution.analyse(raw_product)
    variance = degree_variance(conditioned._coefficient_variance[SCALED])
    for row, limit in enumerate(band_limit(variance).tolist()):
        end = math.ceil(limit * 1.25)
        total = (coeffs[:, :, :, row].abs() ** 2).sum()
        assert (raw_coeffs[:, :, :, row, end:].abs() ** 2).sum() > raw_leakage * total
        assert (coeffs[:, :, :, row, end:].abs() ** 2).sum() < 1e-8 * total


def test_filtering_preserves_the_scaled_variance() -> None:
    conditioned = make_conditioned()
    plain = make_plain()
    inputs = spread_inputs()

    field = advance(conditioned, 0, inputs)[:, :, :, SCALED]
    raw_product = advance(plain, 0)[:, :, :, SCALED] * conditioned.std_modulation(inputs).unsqueeze(1)

    torch.testing.assert_close(area_mean(field**2), area_mean(raw_product**2), rtol=1e-4, atol=1e-6)


def test_scaled_field_decays_by_phi_and_slides_with_the_history() -> None:
    """After the first step the spread is no longer read: the OU process carries the scaling forward."""
    conditioned = make_conditioned()
    first = advance(conditioned, 0, spread_inputs())
    conditioned.noise._draw_normal = lambda out: out.zero_()  # no innovation: only the decay remains

    second = advance(conditioned, 1)

    torch.testing.assert_close(second[:, :, 0], first[:, :, 1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(second[:, :, 1], math.exp(-1.0) * first[:, :, 1], atol=1e-5, rtol=1e-5)


def test_history_slides_in_lockstep_with_innovations() -> None:
    conditioned = make_conditioned()
    previous = advance(conditioned, 0, spread_inputs())
    for fcstep in range(1, 3):
        current = advance(conditioned, fcstep)
        torch.testing.assert_close(current[:, :, 0], previous[:, :, 1], atol=1e-5, rtol=1e-5)
        previous = current


def test_antithetic_pairs_survive_scaling() -> None:
    conditioned = make_conditioned(centered=True)
    field = advance(conditioned, 0, spread_inputs(), ensemble_size=4)

    torch.testing.assert_close(field[:, 0], -field[:, 1])
    torch.testing.assert_close(field[:, 2], -field[:, 3])


def test_scaling_runs_in_fp32_under_autocast() -> None:
    inputs = spread_inputs()
    reference = advance(make_conditioned(), 0, inputs)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        field = advance(make_conditioned(), 0, inputs.to(torch.bfloat16))

    assert field.dtype == torch.float32
    torch.testing.assert_close(field, reference, atol=0.05, rtol=0.05)


def test_smoothing_takes_the_coarsest_of_a_channels_variables() -> None:
    channels = {**CHANNELS, "q": {**CHANNELS["q"], "smoothing_km": 300.0}}
    conditioned = make_conditioned({"smoothing_km": 100.0, "smoothing_km_by_variable": {"2t": 400.0}}, channels)

    # t: the coarser of t_850 (default 100 km) and 2t (400 km); q: its own override
    expected = heat_kernel_response(torch.tensor([kT_from_length_scale(400.0), kT_from_length_scale(300.0)]), NLAT)
    response = conditioned.std_modulation.smooth_response
    torch.testing.assert_close(response, expected[:, : response.shape[-1]].float())


def test_smooth_to_channel_needs_kt_channels() -> None:
    with pytest.raises(ValueError, match="smoothing_km"):
        make_conditioned({"smooth_to_channel": True})
    make_conditioned({"smooth_to_channel": True}, {**CHANNELS, "q": {**CHANNELS["q"], "smoothing_km": 500.0}})


def test_layout_and_first_draw_are_logged(caplog) -> None:
    """The layout and first-draw lines are how a run is checked, so they must format."""
    caplog.set_level(logging.INFO, logger="anemoi.models.layers.ensemble")
    advance(make_conditioned(), 0, spread_inputs())

    messages = "\n".join(caplog.messages)
    assert "2 of 3 channels scaled by 3 'std_' inputs (2 distinct groups); reference=climatology" in messages
    assert "scaled draw; multiplier in" in messages


def test_first_step_requires_the_spread() -> None:
    with pytest.raises(ValueError, match="first rollout step"):
        advance(make_conditioned(), 0)


def test_spread_shape_is_validated() -> None:
    with pytest.raises(ValueError, match="expected"):
        advance(make_conditioned(), 0, spread_inputs()[..., :2])


def test_wrong_normalizer_fails_on_the_first_sample() -> None:
    with pytest.raises(ValueError, match="normalizer"):
        advance(make_conditioned(), 0, spread_inputs() * 1000.0)


@pytest.mark.parametrize(
    "modulation, match",
    [
        ({"normalizer": None}, "needs modulation.normalizer"),
        ({"normalizer": "mean-std"}, "rescales without shifting"),
        ({"reference": "sample_mean", "rescale": "climatology"}, "needs reference 'climatology'"),
        ({"rescale": "sometimes"}, "rescale must be one of"),
        ({"smooth_to_chanel": True}, "unknown keys"),
    ],
)
def test_invalid_modulation_is_rejected(modulation, match) -> None:
    with pytest.raises(ValueError, match=match):
        make_conditioned(modulation)


def test_climatology_needs_the_dataset_statistics() -> None:
    with pytest.raises(ValueError, match="statistics"):
        SphericalInputConditionedNoise(
            grid=NLAT,
            noise=noise_params(),
            channels=CHANNELS,
            modulation=MODULATION,
            name_to_index=STD_NAME_TO_INDEX,
            num_time_steps=N_STEP_INPUT,
        )


def test_scaling_needs_a_channel_with_spread_and_a_diffusion_state() -> None:
    with pytest.raises(ValueError, match="names 'spread'"):
        make_conditioned(channels=unscaled_channels())
    with pytest.raises(ValueError, match="diffusion"):
        make_conditioned(noise={"type": "white"})


def test_spread_buffers_are_not_checkpointed() -> None:
    """Checkpoints of the experiment and its baseline must hold the same keys."""
    assert make_conditioned().state_dict().keys() == make_plain().state_dict().keys()
