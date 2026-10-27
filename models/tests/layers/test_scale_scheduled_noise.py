# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""ScaleScheduledNoiseConditioning: coherent noise on the hidden nodes, its schedule, and the model wiring."""

import math

import pytest
import torch
from torch import nn
from torch_geometric.data import HeteroData

from anemoi.models.layers.ensemble import NoiseConditioning
from anemoi.models.layers.ensemble import ScaleScheduledNoiseConditioning
from anemoi.models.layers.spherical_noise import EARTH_RADIUS_KM
from anemoi.models.models.ens_encoder_processor_decoder import AnemoiEnsModelEncProcDec

GRID_NLAT, LMAX = 64, 48


def _fibonacci_nodes(n: int) -> torch.Tensor:
    """(lat, lon) in radians of ``n`` quasi-uniform points, like the hidden nodes."""
    i = torch.arange(n, dtype=torch.float64) + 0.5
    lat = torch.asin(1 - 2 * i / n)
    lon = torch.remainder(i * math.pi * (3 - math.sqrt(5)), 2 * math.pi)
    return torch.stack([lat, lon], dim=1)


def _graph(coords: torch.Tensor) -> HeteroData:
    graph = HeteroData()
    graph["hidden"].x = coords.float()
    return graph


def _schedule(**overrides) -> dict:
    schedule = {
        "length_scales_km": [300, 1500, 6000],
        "decorrelation_hours": [6, 72, 720],
        "lead_hours": [0, 6, 12],
        "shares": [[70, 20, 10], [20, 50, 30], [0, 20, 80]],
    }
    schedule.update(overrides)
    return schedule


def _injector(coords: torch.Tensor, *, schedule: dict | None = None, noise_std: float = 1.0, **kwargs):
    return ScaleScheduledNoiseConditioning(
        noise_std=noise_std,
        noise_channels_dim=4,
        noise_mlp_hidden_dim=8,
        layer_kernels={},
        schedule=schedule or _schedule(),
        grid_nlat=GRID_NLAT,
        lmax=LMAX,
        graph_data=_graph(coords),
        **kwargs,
    )


def _draw(injector, fcstep: int, *, ensemble_size: int = 32, seed: int = 3, **kwargs) -> torch.Tensor:
    return injector.sample_noise(
        fcstep=fcstep,
        batch_size=1,
        ensemble_size=ensemble_size,
        device=torch.device("cpu"),
        seed=seed,
        **kwargs,
    )[
        0
    ]  # (ensemble, hidden, channels)


def _member_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Correlation over members and channels of two noise tensors (zero mean, as the noise is)."""
    a, b = a.double().flatten(), b.double().flatten()
    return ((a * b).mean() / (a.pow(2).mean() * b.pow(2).mean()).sqrt()).item()


# ------------------------------------------------------------------------------------------------ the noise itself


def test_every_node_gets_unit_variance_noise_at_every_lead_time() -> None:
    """As NoiseConditioning: N(0, noise_std^2) per node and channel, whatever the scales, also near the poles."""
    coords = _fibonacci_nodes(2000)
    injector = _injector(coords, noise_std=2.0)
    polar = coords[:, 0].abs() > math.radians(70)

    for fcstep in range(5):  # through the schedule and past its last row
        noise = _draw(injector, fcstep, ensemble_size=64)
        assert noise.shape == (64, 2000, 4)
        assert abs(noise.mean().item()) < 0.15
        assert noise.var().item() == pytest.approx(4.0, rel=0.06)
        assert noise[:, polar].var().item() == pytest.approx(4.0, rel=0.12)
        assert noise[:, ~polar].var().item() == pytest.approx(4.0, rel=0.06)


def test_large_scale_noise_is_smoother_than_small_scale_noise() -> None:
    """Points 1000 km apart: unrelated at a 300 km scale, almost equal at 4000 km."""
    base = _fibonacci_nodes(600)
    base = base[base[:, 0].abs() < math.radians(60)]
    shifted = base.clone()
    shifted[:, 0] += 1000.0 / EARTH_RADIUS_KM  # 1000 km north
    schedule = {"length_scales_km": [300, 4000], "decorrelation_hours": [6, 6], "lead_hours": [0, 6],
                "shares": [[1, 0], [0, 1]]}  # fmt: skip
    injector = _injector(torch.cat([base, shifted]), schedule=schedule)
    n = len(base)

    small = _draw(injector, 0, ensemble_size=64)
    large = _draw(injector, 1, ensemble_size=64)

    assert abs(_member_correlation(small[:, :n], small[:, n:])) < 0.15
    assert _member_correlation(large[:, :n], large[:, n:]) > 0.85


@pytest.mark.parametrize(
    ("shares", "expected"),
    [
        # Constant shares: the correlation from one step to the next is sum_k w_k phi_k.
        ([[50, 50, 0], [50, 50, 0]], 0.5 * math.exp(-1) + 0.5 * math.exp(-6 / 72)),
        # Shares that change between the steps: sum_k sqrt(w_k(0) w_k(1)) phi_k.
        ([[100, 0, 0], [25, 75, 0]], math.sqrt(0.25) * math.exp(-1)),
    ],
)
def test_each_scale_decorrelates_at_its_own_rate(shares, expected) -> None:
    coords = _fibonacci_nodes(1500)
    injector = _injector(coords, schedule=_schedule(lead_hours=[0, 6], shares=shares))

    first = _draw(injector, 0, ensemble_size=64).clone()
    second = _draw(injector, 1, ensemble_size=64)

    assert _member_correlation(first, second) == pytest.approx(expected, abs=0.05)


def test_a_new_rollout_starts_afresh() -> None:
    coords = _fibonacci_nodes(1500)
    injector = _injector(coords, schedule=_schedule(decorrelation_hours=[720, 720, 720]))
    _draw(injector, 0)
    end_of_rollout = _draw(injector, 2).clone()

    start_of_next = _draw(injector, 0)

    assert abs(_member_correlation(end_of_rollout, start_of_next)) < 0.1


def test_the_channels_are_independent() -> None:
    noise = _draw(_injector(_fibonacci_nodes(1500)), 0, ensemble_size=64)
    for a in range(4):
        for b in range(a + 1, 4):
            assert abs(_member_correlation(noise[..., a], noise[..., b])) < 0.1


# ------------------------------------------------------------------------------------------------ seeds and members


def test_a_seed_sets_the_noise_and_members_differ() -> None:
    coords = _fibonacci_nodes(500)
    one, other = _injector(coords), _injector(coords)

    a = _draw(one, 0, ensemble_size=2, seed=11)
    assert torch.equal(a, _draw(other, 0, ensemble_size=2, seed=11))
    assert not torch.allclose(a, _draw(other, 0, ensemble_size=2, seed=12))
    assert abs(_member_correlation(a[0], a[1])) < 0.2


def test_a_member_draws_the_same_noise_wherever_it_sits() -> None:
    """Global member 1 drawn as the second of two members, or alone on another device."""
    coords = _fibonacci_nodes(500)
    together = _draw(_injector(coords), 0, ensemble_size=2, member_offset=0, num_members_total=2)
    alone = _draw(_injector(coords), 0, ensemble_size=1, member_offset=1, num_members_total=2)

    torch.testing.assert_close(together[1], alone[0])


# ------------------------------------------------------------------------------------------------ the schedule


def test_shares_are_interpolated_between_lead_times_and_held_after_the_last() -> None:
    injector = _injector(_fibonacci_nodes(100), timestep_hours=3)
    rows = torch.tensor([[70, 20, 10], [20, 50, 30], [0, 20, 80]], dtype=torch.float64) / 100

    torch.testing.assert_close(injector.shares_at(0), rows[0])
    torch.testing.assert_close(injector.shares_at(1), (rows[0] + rows[1]) / 2)  # +3 h
    torch.testing.assert_close(injector.shares_at(2), rows[1])
    torch.testing.assert_close(injector.shares_at(40), rows[2])


def test_shares_are_normalised_per_lead_time() -> None:
    injector = _injector(_fibonacci_nodes(100), schedule=_schedule(lead_hours=[0], shares=[[2, 1, 1]]))
    torch.testing.assert_close(injector.shares_at(7), torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"decorrelation_hours": [6, 72]}, "one decorrelation_hours per length scale"),
        ({"lead_hours": [6, 12, 18]}, "start at 0 and increase"),
        ({"lead_hours": [0, 12, 6]}, "start at 0 and increase"),
        ({"shares": [[70, 20, 10], [20, 50, 30]]}, "one row per lead time"),
        ({"shares": [[70, 20], [20, 50], [0, 20]]}, "one row per lead time"),
        ({"shares": [[70, 20, 10], [-1, 50, 30], [0, 20, 80]]}, "non-negative"),
        ({"shares": [[70, 20, 10], [0, 0, 0], [0, 20, 80]]}, "non-negative"),
        ({"length_scales_km": [300, 0, 6000]}, "must be positive"),
    ],
)
def test_bad_schedules_are_rejected(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        _injector(_fibonacci_nodes(100), schedule=_schedule(**overrides))


def test_a_schedule_needs_all_its_keys() -> None:
    schedule = _schedule()
    del schedule["shares"]
    with pytest.raises(ValueError, match="exactly the keys"):
        _injector(_fibonacci_nodes(100), schedule=schedule)


# ------------------------------------------------------------------------------------------------ the grid


def test_each_node_takes_its_nearest_grid_point() -> None:
    coords = torch.cat([_fibonacci_nodes(3000), torch.tensor([[math.pi / 2, 0.3], [-math.pi / 2, 1.0]])])
    index = ScaleScheduledNoiseConditioning._nearest_grid_points(coords[:, 0], coords[:, 1], 16)

    from anemoi.models.layers.spectral_helpers import legendre_gauss_weights

    nodes, _ = legendre_gauss_weights(16)
    ring_lat = math.pi / 2 - torch.flip(torch.from_numpy(nodes.copy()).acos(), dims=[0])
    grid_lat = ring_lat.repeat_interleave(32)
    grid_lon = (torch.arange(32, dtype=torch.float64) * 2 * math.pi / 32).repeat(16)

    def cosine(lat, lon, glat, glon):
        return torch.sin(lat) * torch.sin(glat) + torch.cos(lat) * torch.cos(glat) * torch.cos(lon - glon)

    brute = cosine(coords[:, :1], coords[:, 1:], grid_lat[None], grid_lon[None]).max(dim=1).values
    chosen = cosine(coords[:, 0], coords[:, 1], grid_lat[index], grid_lon[index])
    torch.testing.assert_close(chosen, brute)


def test_the_grid_matches_the_transform() -> None:
    """A degree-1 field from the transform, read at the nodes, is sin(lat) and cos(lat) cos(lon)."""
    coords = _fibonacci_nodes(2000)
    injector = _injector(coords)
    isht = injector._transform[0]
    lat, lon = coords[:, 0], coords[:, 1]

    for m, value, expected in [
        (0, 1.0, torch.sin(lat)),
        (1, 1.0, torch.cos(lat) * torch.cos(lon)),
        (1, 1.0j, -torch.cos(lat) * torch.sin(lon)),
    ]:
        coefficients = torch.zeros(LMAX, LMAX, dtype=torch.complex64)
        coefficients[1, m] = value
        field = isht(coefficients[None])[0][injector.hidden_grid_index].double()
        assert _member_correlation(field, expected) > 0.995


def test_a_projection_matrix_is_not_supported() -> None:
    with pytest.raises(ValueError, match="not supported"):
        _injector(_fibonacci_nodes(10), noise_matrix="some_file.npz")


def test_the_hidden_nodes_are_needed() -> None:
    with pytest.raises(ValueError, match="'hidden' nodes"):
        ScaleScheduledNoiseConditioning(
            noise_std=1, noise_channels_dim=4, noise_mlp_hidden_dim=8, layer_kernels={}, schedule=_schedule()
        )


def test_forward_gives_the_conditioning_noise_condition_layer_norm_expects() -> None:
    """Same output as NoiseConditioning: (batch * ensemble * hidden, channels), through the same MLP."""
    coords = _fibonacci_nodes(300)
    injector = _injector(coords)
    x = torch.zeros(2 * 3 * 300, 16)

    out_x, condition = injector(x=x, batch_size=2, ensemble_size=3, grid_size=300, grid_shard_sizes=None, fcstep=1)

    assert out_x is x
    assert condition.shape == (2 * 3 * 300, 4)
    assert isinstance(injector, NoiseConditioning)
    with pytest.raises(ValueError, match="hidden nodes"):
        injector(x=x, batch_size=2, ensemble_size=3, grid_size=299, grid_shard_sizes=None)


# ------------------------------------------------------------------------------------------------ model wiring


class _InjectorReached(RuntimeError):
    pass


class _CaptureInjector(nn.Module):
    """Records what the model passes the noise injector, then stops the forward."""

    def __init__(self, uses_rollout_state: bool) -> None:
        super().__init__()
        self.uses_rollout_state = uses_rollout_state

    def forward(self, **kwargs):
        self.kwargs = kwargs
        raise _InjectorReached


class _HiddenAttributes(nn.Module):
    num_nodes = {"hidden": 3}

    def forward(self, node_name: str, batch_size: int) -> torch.Tensor:
        return torch.zeros(3, 4)


class _GraphProvider(nn.Module):
    def get_edges(self, **kwargs):
        return None, None, None


class _Encoder(nn.Module):
    def forward(self, x, **kwargs):
        return x[0], torch.zeros(3, 4)


class _StubEnsModel(AnemoiEnsModelEncProcDec):
    """One ERA5 encoder, up to the noise injector."""

    def __init__(self, injector: nn.Module) -> None:
        nn.Module.__init__(self)
        self.input_datasets = ["era5"]
        self.dataset2encoder = {"era5": "era5"}
        self.first_step_only_datasets = set()
        self._graph_name_hidden = "hidden"
        self.input_noise = None
        self.input_noise_dataset = None
        self.node_attributes = _HiddenAttributes()
        self.encoder_graph_provider = nn.ModuleDict({"era5": _GraphProvider()})
        self.encoder = nn.ModuleDict({"era5": _Encoder()})
        self.latent_aggregator = lambda hidden, latents: latents["era5"]
        self.noise_injector = injector

    def _resolve_in_out_sharded(self, dataset_names, grid_shard_sizes):
        return dict.fromkeys(dataset_names, False)

    def _assert_valid_sharding(self, *args, **kwargs) -> None:
        return None

    def _assemble_input(self, x, fcstep, batch_ens_size, grid_shard_sizes=None, model_comm_group=None, **kwargs):
        return torch.ones(3, 4), None, None


def _injector_kwargs(uses_rollout_state: bool, **forward_kwargs) -> dict:
    model = _StubEnsModel(_CaptureInjector(uses_rollout_state))
    with pytest.raises(_InjectorReached):
        model({"era5": torch.zeros(1, 2, 1, 3, 2)}, **forward_kwargs)
    return model.noise_injector.kwargs


@pytest.mark.parametrize("fcstep", [0, 1, 7, 39])
def test_the_model_passes_the_unclamped_rollout_step_and_the_member_layout(fcstep) -> None:
    kwargs = _injector_kwargs(
        True,
        fcstep=fcstep,
        ensemble_member_offset=4,
        ensemble_members_total=8,
        ensemble_group_id=2,
        input_noise_seed=99,
    )
    assert kwargs["fcstep"] == fcstep  # not the encoder's min(1, fcstep)
    assert (kwargs["member_offset"], kwargs["num_members_total"], kwargs["group_id"], kwargs["seed"]) == (4, 8, 2, 99)


def test_other_injectors_are_called_as_before() -> None:
    kwargs = _injector_kwargs(False, fcstep=5, ensemble_member_offset=4, input_noise_seed=99)
    assert set(kwargs) == {"x", "batch_size", "ensemble_size", "grid_size", "grid_shard_sizes", "model_comm_group"}
