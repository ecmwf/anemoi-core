# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Compare grid-sharded spectral losses with a single-process reference on CPU."""

import importlib.util
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.spectral_transforms import octahedral_lons_per_lat
from anemoi.training.losses import SpectralAMSELoss
from anemoi.training.losses import SpectralCRPSLoss
from anemoi.training.losses.base import BaseLoss

NLAT = 8
OCTAHEDRAL_POINTS = sum(octahedral_lons_per_lat(NLAT))
REGULAR_POINTS = NLAT * 2 * NLAT
CARTESIAN = {"x_dim": 8, "y_dim": 13}
CARTESIAN_POINTS = CARTESIAN["x_dim"] * CARTESIAN["y_dim"]
N_TIME = 2
N_MEMBERS = 3
GRID_AXIS = 3
# Fewer, as many as and more variables than the 2 or 3 ranks, so that some ranks get no variable when the
# variables are shared out for the transforms.
VARIABLE_COUNTS = (1, 2, 4)

# Each loss with its number of grid points and ensemble members.
LOSSES: dict[str, tuple[Callable[[], BaseLoss], int, int]] = {
    "spectral CRPS, octahedral, ring FFTs": (
        lambda: SpectralCRPSLoss(transform="octahedral_sht", nlat=NLAT),
        OCTAHEDRAL_POINTS,
        N_MEMBERS,
    ),
    "spectral CRPS, octahedral, padded FFTs": (
        lambda: SpectralCRPSLoss(transform="octahedral_sht", nlat=NLAT, fft_backend="padded"),
        OCTAHEDRAL_POINTS,
        N_MEMBERS,
    ),
    "spectral CRPS, regular": (
        lambda: SpectralCRPSLoss(transform="regular_sht", nlat=NLAT, nlon=2 * NLAT),
        REGULAR_POINTS,
        N_MEMBERS,
    ),
    "spectral AMSE, octahedral, ring FFTs": (
        lambda: SpectralAMSELoss(transform="octahedral_sht", nlat=NLAT),
        OCTAHEDRAL_POINTS,
        1,
    ),
    "spectral CRPS, fft2d": (
        lambda: SpectralCRPSLoss(transform="fft2d", **CARTESIAN),
        CARTESIAN_POINTS,
        N_MEMBERS,
    ),
    "spectral AMSE, fft2d": (
        lambda: SpectralAMSELoss(transform="fft2d", **CARTESIAN),
        CARTESIAN_POINTS,
        1,
    ),
}
if importlib.util.find_spec("torch_dct") is not None:
    LOSSES["spectral CRPS, dct2d"] = (
        lambda: SpectralCRPSLoss(transform="dct2d", **CARTESIAN),
        CARTESIAN_POINTS,
        N_MEMBERS,
    )


def _inputs(n_grid: int, n_members: int, n_variables: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(n_variables)
    prediction = torch.randn(1, N_TIME, n_members, n_grid, n_variables, generator=generator)
    target = torch.randn(1, N_TIME, 1, n_grid, n_variables, generator=generator)
    return prediction, target


def _loss_and_gradient(
    loss: BaseLoss,
    prediction: torch.Tensor,
    target: torch.Tensor,
    group: dist.ProcessGroup | None,
) -> tuple[torch.Tensor, torch.Tensor, slice]:
    """Evaluate the loss on this rank's part of the grid, or on the whole grid without a group."""
    if group is None:
        local, kwargs = slice(None), {}
    else:
        shard_sizes = get_shard_sizes(prediction, GRID_AXIS, model_comm_group=group)
        start = sum(shard_sizes[: group.rank()])
        local = slice(start, start + shard_sizes[group.rank()])
        kwargs = {"grid_shard_slice": local, "group": group, "grid_shard_sizes": shard_sizes, "grid_dim": -2}
    local_prediction = prediction[:, :, :, local].clone().requires_grad_()
    value = loss(local_prediction, target[:, :, :, local].clone(), **kwargs)
    value.backward()
    return value.detach(), local_prediction.grad, local


def _check_sharded_losses(rank: int, world_size: int, init_file: str) -> None:
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        group = dist.new_group(list(range(world_size)))
        mismatches = []
        # Every rank runs every case before any check fails, so no rank is left waiting in a
        # collective that the others never reach.
        for name, (build, n_grid, n_members) in LOSSES.items():
            for n_variables in VARIABLE_COUNTS:
                case = f"{name}, {n_variables} variable(s)"
                prediction, target = _inputs(n_grid, n_members, n_variables)
                expected, expected_grad, _ = _loss_and_gradient(build(), prediction, target, None)
                actual, actual_grad, local = _loss_and_gradient(build(), prediction, target, group)
                if not torch.allclose(actual, expected, rtol=1e-10, atol=1e-12):
                    mismatches.append(f"{case}: loss {actual} differs from {expected}")
                if actual_grad is None:
                    mismatches.append(f"{case}: the prediction gets no gradient")
                elif not torch.allclose(actual_grad, expected_grad[:, :, :, local], rtol=1e-10, atol=1e-12):
                    mismatches.append(f"{case}: gradient differs")
        assert not mismatches, f"rank {rank} of {world_size}:\n" + "\n".join(mismatches)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Requires Gloo")
@pytest.mark.parametrize("world_size", [2, 3])
def test_grid_sharded_spectral_losses_match_single_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
) -> None:
    """Spectral losses on a grid split over ranks give the single-process loss and gradients.

    The grid splits unevenly over 3 ranks, and some cases have fewer variables than ranks. A rank without
    variables transforms no fields and still takes part in the exchanges of the forward and backward pass.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    mp.spawn(_check_sharded_losses, args=(world_size, str(tmp_path / "rendezvous")), nprocs=world_size, join=True)
