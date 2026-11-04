# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import DictConfig
from pydantic import TypeAdapter
from pydantic import ValidationError

from anemoi.models.layers.spectral_helpers import InverseSphericalHarmonicTransform
from anemoi.models.layers.spectral_transforms import octahedral_lons_per_lat
from anemoi.training.losses import WaveletEnergyScoreLoss
from anemoi.training.losses import get_loss_function
from anemoi.training.schemas.training import LossSchemas
from anemoi.training.schemas.training import WaveletEnergyScoreLossSchema
from anemoi.training.utils.enums import TensorDim

NLAT = 24
TRUNCATION = 11  # five scales: l = 1, 2-3, 3-7, 5-11 and 9-11
WEIGHTS = [0.0, 0.0, 1.0, 2.0, 0.0]
N_POINTS = sum(octahedral_lons_per_lat(NLAT))


def band_limited(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    """Random fields [..., grid] that the octahedral grid represents exactly up to TRUNCATION."""
    generator = torch.Generator().manual_seed(seed)
    coefficients = torch.randn(*shape, TRUNCATION + 1, TRUNCATION + 1, dtype=torch.complex128, generator=generator)
    coefficients = coefficients.tril()
    coefficients[..., 0] = coefficients[..., 0].real
    inverse = InverseSphericalHarmonicTransform(octahedral_lons_per_lat(NLAT), TRUNCATION)
    return inverse(coefficients).float()


def make_inputs(members: int = 3, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Prediction [1, 1, members, grid, 2 variables] and target [1, 1, 1, grid, 2 variables]."""
    pred = band_limited((1, 1, members, 2), seed).movedim(-2, -1)
    target = band_limited((1, 1, 1, 2), seed + 1).movedim(-2, -1)
    return pred, target


def make_loss(**kwargs) -> WaveletEnergyScoreLoss:
    kwargs.setdefault("scale_weights", WEIGHTS)
    kwargs.setdefault("nlat", NLAT)
    return WaveletEnergyScoreLoss(transform="octahedral_sht", smoothing_octaves=2, **kwargs)


def reference_scores(loss: WaveletEnergyScoreLoss, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The weighted energy scores [variable, grid], written out over all ordered pairs of distinct members."""
    structure = loss.structure
    members = pred[0, 0].movedim(-1, 0)  # [variable, member, grid]
    truth = target[0, 0, 0].movedim(-1, 0)  # [variable, grid]
    n = members.shape[1]
    total = torch.zeros(2, N_POINTS)
    for k, weight in enumerate(loss.weights):
        v_members = loss.structure_vectors(structure.tensor(structure.coefficients(members), k))
        v_truth = loss.structure_vectors(structure.tensor(structure.coefficients(truth), k))
        skill = sum((v_members[:, i] - v_truth).norm(dim=-2) for i in range(n)) / n
        spread = sum((v_members[:, i] - v_members[:, h]).norm(dim=-2) for i in range(n) for h in range(n) if i != h)
        pairs = n * (n - 1) if loss.fair else n * n
        total += weight * (skill - spread / (2 * pairs))
    return total


@pytest.mark.parametrize("fair", [True, False])
def test_scores_are_the_weighted_energy_scores_of_the_structure_vectors(fair: bool) -> None:
    pred, target = make_inputs()
    loss = make_loss(fair=fair)
    assert loss.structure.scales == [2, 3]
    expected = reference_scores(loss, pred, target)

    torch.testing.assert_close(loss.point_scores(pred, target)[0, 0, 0].T, expected, rtol=1e-5, atol=1e-6)
    # Without scalers the loss adds up the grid points, variable by variable.
    torch.testing.assert_close(loss(pred, target, squash=False), expected.sum(-1), rtol=1e-5, atol=1e-5)


def test_structure_vectors_from_eigenvalues_and_direction() -> None:
    """J with eigenvalues a >= b, the larger along the angle theta from east.

    It gives v = (sqrt(a + b), (a - b) / sqrt(a + b + floor) (cos 2 theta, sin 2 theta)).
    """
    loss = make_loss(direction_floor=0.3)
    generator = torch.Generator().manual_seed(4)
    larger, smaller = torch.rand(2, 2, N_POINTS, dtype=torch.float64, generator=generator).sort(dim=0).values.flip(0)
    theta = torch.pi * torch.rand(2, N_POINTS, dtype=torch.float64, generator=generator)
    cos, sin = theta.cos(), theta.sin()
    j_ee = larger * cos**2 + smaller * sin**2
    j_nn = larger * sin**2 + smaller * cos**2
    j_en = (larger - smaller) * sin * cos

    energy = larger + smaller
    area = loss.structure.area
    floor = 0.3 * (energy * area).sum(-1, keepdim=True)  # the area adds up to 1
    direction = (larger - smaller) / (energy + floor).sqrt()
    expected = torch.stack([energy.sqrt(), direction * (2 * theta).cos(), direction * (2 * theta).sin()], dim=-2)
    result = loss.structure_vectors(torch.stack([j_ee, j_nn, j_en], dim=-2))
    torch.testing.assert_close(result, expected, rtol=1e-9, atol=1e-12)


def test_the_score_scales_with_the_fields() -> None:
    """The floor scales with each field's energy, so scaling all fields scales every vector and the score."""
    pred, target = make_inputs()
    loss = make_loss()
    torch.testing.assert_close(
        loss(3.0 * pred, 3.0 * target, squash=False),
        3.0 * loss(pred, target, squash=False),
        rtol=1e-4,
        atol=1e-6,
    )


def test_a_perfect_ensemble_scores_zero() -> None:
    _, target = make_inputs()
    loss = make_loss()
    perfect = target.expand(-1, -1, 3, -1, -1)
    # Only the tiny amount under each square root is left.
    assert loss.point_scores(perfect, target).abs().max() < 1e-5


def test_gradients_are_finite_for_zero_fields_and_identical_members() -> None:
    pred = torch.zeros(1, 1, 3, N_POINTS, 1, requires_grad=True)
    target = torch.zeros(1, 1, 1, N_POINTS, 1)
    make_loss()(pred, target).backward()
    assert torch.isfinite(pred.grad).all()


def test_needs_one_weight_per_scale() -> None:
    with pytest.raises(ValueError, match="one weight per wavelet scale"):
        make_loss(scale_weights=[1.0, 1.0])


def test_needs_a_weight_above_zero() -> None:
    with pytest.raises(ValueError, match="at least one above 0"):
        make_loss(scale_weights=[0.0, 0.0, 0.0, 0.0, 0.0])


def test_rejects_missing_values() -> None:
    with pytest.raises(ValueError, match="ignore_nans must be False"):
        make_loss(ignore_nans=True)


def test_rejects_autocast() -> None:
    with pytest.raises(ValueError, match="no_autocast must be True"):
        make_loss(no_autocast=False)


def test_rejects_too_few_longitudes_on_a_regular_grid() -> None:
    # Scale 3 at T11: the products of its Riesz pair reach order 24, and its smoothing degree 10.
    with pytest.raises(ValueError, match="need more than 34 longitudes"):
        WaveletEnergyScoreLoss(
            transform="regular_sht",
            nlat=NLAT,
            nlon=34,
            scale_weights=WEIGHTS,
            smoothing_octaves=2,
        )


def test_rejects_an_odd_number_of_octahedral_latitudes() -> None:
    with pytest.raises(ValueError, match="even number of latitudes"):
        make_loss(nlat=NLAT - 1)


def test_rejects_a_grid_of_the_wrong_size() -> None:
    pred, target = make_inputs()
    with pytest.raises(ValueError, match="grid points"):
        make_loss()(pred[..., :-1, :], target[..., :-1, :])


def test_builds_from_a_validated_config() -> None:
    config = {
        "_target_": "anemoi.training.losses.WaveletEnergyScoreLoss",
        "transform": "octahedral_sht",
        "nlat": NLAT,
        "scale_weights": WEIGHTS,
        "smoothing_octaves": 2,
        "scalers": [],
    }
    assert isinstance(TypeAdapter(LossSchemas).validate_python(config), WaveletEnergyScoreLossSchema)
    loss = get_loss_function(DictConfig(config))
    assert isinstance(loss, WaveletEnergyScoreLoss)
    pred, target = make_inputs()
    assert torch.isfinite(loss(pred, target))


@pytest.mark.parametrize(
    "config",
    [
        {"transform": "octahedral_sht", "nlat": NLAT, "scale_weights": [0.0, 0.0]},  # no weight above 0
        {"transform": "reduced_sht", "scale_weights": [1.0]},  # no grid name
        {"transform": "reduced_sht", "grid": "n640", "scale_weights": [1.0]},  # an unsupported grid name
        {"transform": "octahedral_sht", "scale_weights": [1.0]},  # no nlat
        {"transform": "octahedral_sht", "nlat": NLAT, "scale_weights": [1.0], "no_autocast": False},
        {"transform": "octahedral_sht", "nlat": NLAT, "scale_weights": [1.0], "ignore_nans": True},
    ],
)
def test_schema_rejects_invalid_configs(config: dict) -> None:
    with pytest.raises(ValidationError):
        WaveletEnergyScoreLossSchema(_target_="anemoi.training.losses.WaveletEnergyScoreLoss", **config)


def _check_sharded_grid(rank: int, world_size: int, n_variables: int, init_file: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        pred, target = make_inputs()
        pred = torch.cat([pred, 2.0 * pred[..., :1]], dim=-1)[..., :n_variables]
        target = torch.cat([target, 2.0 * target[..., :1]], dim=-1)[..., :n_variables]
        node_weights = torch.rand(N_POINTS, generator=torch.Generator().manual_seed(3))

        loss = make_loss()
        loss.add_scaler(TensorDim.GRID, node_weights, name="node_weights")
        full_pred = pred.clone().requires_grad_()
        expected = loss(full_pred, target)
        expected.backward()

        # Uneven grid shards: all but the last rank hold 7 points fewer than an even split.
        grid_shard_sizes = [N_POINTS // world_size - 7] * (world_size - 1)
        grid_shard_sizes.append(N_POINTS - sum(grid_shard_sizes))
        start = sum(grid_shard_sizes[:rank])
        shard = slice(start, start + grid_shard_sizes[rank])
        local_pred = pred[..., shard, :].clone().requires_grad_()
        result = loss(
            local_pred,
            target[..., shard, :],
            grid_shard_slice=shard,
            group=dist.group.WORLD,
            grid_shard_sizes=grid_shard_sizes,
            grid_dim=-2,
        )
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)
        result.backward()
        # The reshuffles send each rank the gradient of all variables at its own grid points.
        torch.testing.assert_close(local_pred.grad, full_pred.grad[..., shard, :], rtol=1e-4, atol=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Requires Gloo")
@pytest.mark.parametrize(
    ("world_size", "n_variables"),
    [(2, 3), (3, 2)],
    ids=["variables split unevenly", "a rank without variables"],
)
def test_a_sharded_grid_gives_the_unsharded_loss_and_gradients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
    n_variables: int,
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    mp.spawn(
        _check_sharded_grid,
        args=(world_size, n_variables, str(tmp_path / "rendezvous")),
        nprocs=world_size,
        join=True,
    )
