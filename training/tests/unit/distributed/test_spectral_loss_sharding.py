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

from anemoi.models.data import TensorLayout
from anemoi.models.data.sources import GriddedSource
from anemoi.models.data.sources.base import Source
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.spectral_transforms import octahedral_lons_per_lat
from anemoi.training.losses import FourierCorrelationLoss
from anemoi.training.losses import LogSpectralDistance
from anemoi.training.losses import PowerSpectrumLoss
from anemoi.training.losses import SpectralAMSELoss
from anemoi.training.losses import SpectralCRPSLoss
from anemoi.training.losses.base import BaseLoss
from anemoi.training.utils.enums import TensorDim

WORLD_SIZE = 2
N_VARIABLES = 3
OCTAHEDRAL = {"transform": "octahedral_sht", "nlat": 8, "truncation": 3}
REFERENCE_OCTAHEDRAL_POINTS = 2 * (20 + 24 + 28 + 32)
FFT2D = {"transform": "fft2d", "x_dim": 4, "y_dim": 4}
LAYOUT = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)

NLAT = 8
OCTAHEDRAL_POINTS = sum(octahedral_lons_per_lat(NLAT))
REGULAR_POINTS = NLAT * 2 * NLAT
CARTESIAN = {"x_dim": 8, "y_dim": 13}
CARTESIAN_POINTS = CARTESIAN["x_dim"] * CARTESIAN["y_dim"]
# Length of the spectral dimension of the losses. The SHTs on NLAT = 8 latitudes have truncation 3, so
# 16 (l, m) entries or 4 degrees. The 8 x 13 Cartesian grid has 104 coefficients in 8 radial bands.
SHT_MODES, SHT_DEGREES = 16, 4
CARTESIAN_MODES, CARTESIAN_BANDS = CARTESIAN_POINTS, 8
N_TIME = 2
N_MEMBERS = 3
# Fewer, as many as and more variables than the 2 or 3 ranks, so that some ranks get no variable when the
# variables are shared out for the transforms.
VARIABLE_COUNTS = (1, 2, 4)

# Keep the original uneven two-rank regression alongside the expanded backend matrix.
CASES = [
    (SpectralCRPSLoss, OCTAHEDRAL, REFERENCE_OCTAHEDRAL_POINTS, 4, 16),
    (LogSpectralDistance, OCTAHEDRAL, REFERENCE_OCTAHEDRAL_POINTS, 1, 16),
    (FourierCorrelationLoss, OCTAHEDRAL, REFERENCE_OCTAHEDRAL_POINTS, 1, 16),
    (PowerSpectrumLoss, OCTAHEDRAL, REFERENCE_OCTAHEDRAL_POINTS, 1, 4),
    (SpectralAMSELoss, OCTAHEDRAL, REFERENCE_OCTAHEDRAL_POINTS, 1, 4),
    (SpectralCRPSLoss, FFT2D, 16, 4, 16),
    (LogSpectralDistance, FFT2D, 16, 1, 16),
    (FourierCorrelationLoss, FFT2D, 16, 1, 16),
]

# Each loss with its number of grid points, ensemble members and length of the spectral dimension.
LOSSES: dict[str, tuple[Callable[[], BaseLoss], int, int, int]] = {
    "spectral CRPS, octahedral, ring FFTs": (
        lambda: SpectralCRPSLoss(transform="octahedral_sht", nlat=NLAT),
        OCTAHEDRAL_POINTS,
        N_MEMBERS,
        SHT_MODES,
    ),
    "spectral CRPS, octahedral, padded FFTs": (
        lambda: SpectralCRPSLoss(transform="octahedral_sht", nlat=NLAT, fft_backend="padded"),
        OCTAHEDRAL_POINTS,
        N_MEMBERS,
        SHT_MODES,
    ),
    "spectral CRPS, regular": (
        lambda: SpectralCRPSLoss(transform="regular_sht", nlat=NLAT, nlon=2 * NLAT),
        REGULAR_POINTS,
        N_MEMBERS,
        SHT_MODES,
    ),
    "log spectral distance, octahedral": (
        lambda: LogSpectralDistance(transform="octahedral_sht", nlat=NLAT),
        OCTAHEDRAL_POINTS,
        1,
        SHT_MODES,
    ),
    "Fourier correlation, octahedral": (
        lambda: FourierCorrelationLoss(transform="octahedral_sht", nlat=NLAT),
        OCTAHEDRAL_POINTS,
        1,
        SHT_MODES,
    ),
    "power spectrum, octahedral": (
        lambda: PowerSpectrumLoss(transform="octahedral_sht", nlat=NLAT),
        OCTAHEDRAL_POINTS,
        1,
        SHT_DEGREES,
    ),
    "spectral AMSE, octahedral, ring FFTs": (
        lambda: SpectralAMSELoss(transform="octahedral_sht", nlat=NLAT),
        OCTAHEDRAL_POINTS,
        1,
        SHT_DEGREES,
    ),
    "spectral CRPS, fft2d": (
        lambda: SpectralCRPSLoss(transform="fft2d", **CARTESIAN),
        CARTESIAN_POINTS,
        N_MEMBERS,
        CARTESIAN_MODES,
    ),
    "log spectral distance, fft2d": (
        lambda: LogSpectralDistance(transform="fft2d", **CARTESIAN),
        CARTESIAN_POINTS,
        1,
        CARTESIAN_MODES,
    ),
    "Fourier correlation, fft2d": (
        lambda: FourierCorrelationLoss(transform="fft2d", **CARTESIAN),
        CARTESIAN_POINTS,
        1,
        CARTESIAN_MODES,
    ),
    "spectral AMSE, fft2d": (
        lambda: SpectralAMSELoss(transform="fft2d", **CARTESIAN),
        CARTESIAN_POINTS,
        1,
        CARTESIAN_BANDS,
    ),
}
if importlib.util.find_spec("torch_dct") is not None:
    LOSSES["spectral CRPS, dct2d"] = (
        lambda: SpectralCRPSLoss(transform="dct2d", **CARTESIAN),
        CARTESIAN_POINTS,
        N_MEMBERS,
        CARTESIAN_MODES,
    )


def _view(data: torch.Tensor) -> Source:
    """Wrap a (batch, time, ensemble, grid, variables) tensor in the source view the loss takes."""
    return GriddedSource(
        name="data",
        values=data,
        variables=[f"v{i}" for i in range(data.shape[-1])],
        statistics={},
        coordinates=torch.zeros(data.shape[-2], 2, device=data.device),
        layout=LAYOUT,
    )


def _uneven_grid_shard_sizes(n_points: int) -> list[int]:
    first = n_points // 2 + 3
    return [first, n_points - first]


def _build(build: Callable[[], BaseLoss], n_spectral: int) -> BaseLoss:
    """Build the loss with a different weight per spectral entry, so each rank must scale its own modes."""
    loss = build()
    loss.add_scaler(TensorDim.GRID, torch.linspace(0.5, 1.5, n_spectral), name="spectral")
    return loss


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
    """Evaluate the loss per variable on this rank's part of the grid, or on the whole grid without a group."""
    if group is None:
        local, kwargs = slice(None), {}
    else:
        grid_dim = LAYOUT.axis(TensorDim.GRID)
        shard_sizes = get_shard_sizes(prediction, grid_dim, model_comm_group=group)
        start = sum(shard_sizes[: group.rank()])
        local = slice(start, start + shard_sizes[group.rank()])
        kwargs = {"grid_shard_slice": local, "group": group, "grid_shard_sizes": shard_sizes, "grid_dim": grid_dim}
    local_prediction = prediction[:, :, :, local].clone().requires_grad_()
    value = loss(_view(local_prediction), _view(target[:, :, :, local].clone()), squash=False, **kwargs)
    value.sum().backward()
    return value.detach(), local_prediction.grad, local


def _check_sharded_spectral_losses(rank: int, init_file: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(seconds=60),
    )
    try:
        for loss_cls, transform_kwargs, n_points, ensemble_size, n_spectral in CASES:
            loss = loss_cls(**transform_kwargs)
            # A different weight per spectral entry checks that each rank scales its own modes.
            loss.add_scaler(
                TensorDim.GRID,
                torch.linspace(0.5, 1.5, n_spectral, dtype=torch.float64),
                name="spectral",
            )

            # All ranks build the same inputs and keep their own grid points.
            generator = torch.Generator().manual_seed(0)
            pred = torch.randn(1, 2, ensemble_size, n_points, N_VARIABLES, dtype=torch.float64, generator=generator)
            target = torch.randn(1, 2, 1, n_points, N_VARIABLES, dtype=torch.float64, generator=generator)
            grid_shard_sizes = _uneven_grid_shard_sizes(n_points)
            start = sum(grid_shard_sizes[:rank])
            grid_shard_slice = slice(start, start + grid_shard_sizes[rank])

            unsharded_pred = pred.clone().requires_grad_()
            expected = loss(_view(unsharded_pred), _view(target), squash=False)
            expected.sum().backward()

            # Each rank's gradient should match the unsharded gradient on its own grid points.
            sharded_pred = pred[:, :, :, grid_shard_slice].clone().requires_grad_()
            actual = loss(
                _view(sharded_pred),
                _view(target[:, :, :, grid_shard_slice]),
                squash=False,
                grid_shard_slice=grid_shard_slice,
                group=dist.group.WORLD,
                grid_dim=LAYOUT.axis(TensorDim.GRID),
                grid_shard_sizes=grid_shard_sizes,
            )
            actual.sum().backward()

            msg = f"rank={rank}, loss={loss_cls.__name__}, transform={transform_kwargs['transform']}"
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12, msg=msg)
            torch.testing.assert_close(
                sharded_pred.grad,
                unsharded_pred.grad[:, :, :, grid_shard_slice],
                rtol=1e-12,
                atol=1e-12,
                msg=msg,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Requires Gloo")
def test_sharded_spectral_losses_and_gradients_match_unsharded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    mp.spawn(
        _check_sharded_spectral_losses,
        args=(str(tmp_path / "rendezvous"),),
        nprocs=WORLD_SIZE,
        join=True,
    )


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
        for name, (build, n_grid, n_members, n_spectral) in LOSSES.items():
            for n_variables in VARIABLE_COUNTS:
                case = f"{name}, {n_variables} variable(s)"
                prediction, target = _inputs(n_grid, n_members, n_variables)
                expected, expected_grad, _ = _loss_and_gradient(_build(build, n_spectral), prediction, target, None)
                actual, actual_grad, local = _loss_and_gradient(_build(build, n_spectral), prediction, target, group)
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
    Each loss has a different weight per spectral entry, which every rank applies to its own modes.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    mp.spawn(_check_sharded_losses, args=(world_size, str(tmp_path / "rendezvous")), nprocs=world_size, join=True)
