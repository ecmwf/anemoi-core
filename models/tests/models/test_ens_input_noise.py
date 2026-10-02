# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from anemoi.models.layers.ensemble import SphericalInputNoise
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
