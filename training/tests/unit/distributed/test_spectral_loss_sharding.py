# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Check that grid-sharded spectral losses and gradients match the unsharded ones on CPU."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from anemoi.training.losses import FourierCorrelationLoss
from anemoi.training.losses import LogSpectralDistance
from anemoi.training.losses import PowerSpectrumLoss
from anemoi.training.losses import SpectralAMSELoss
from anemoi.training.losses import SpectralCRPSLoss
from anemoi.training.utils.enums import TensorDim

WORLD_SIZE = 2
N_VARIABLES = 3
OCTAHEDRAL = {"transform": "octahedral_sht", "nlat": 8, "truncation": 3}
OCTAHEDRAL_POINTS = 2 * (20 + 24 + 28 + 32)
FFT2D = {"transform": "fft2d", "x_dim": 4, "y_dim": 4}

# (loss class, transform settings, grid points, ensemble size, length of the spectral dimension)
CASES = [
    (SpectralCRPSLoss, OCTAHEDRAL, OCTAHEDRAL_POINTS, 4, 16),
    (LogSpectralDistance, OCTAHEDRAL, OCTAHEDRAL_POINTS, 1, 16),
    (FourierCorrelationLoss, OCTAHEDRAL, OCTAHEDRAL_POINTS, 1, 16),
    (PowerSpectrumLoss, OCTAHEDRAL, OCTAHEDRAL_POINTS, 1, 4),
    (SpectralAMSELoss, OCTAHEDRAL, OCTAHEDRAL_POINTS, 1, 4),
    (SpectralCRPSLoss, FFT2D, 16, 4, 16),
    (LogSpectralDistance, FFT2D, 16, 1, 16),
    (FourierCorrelationLoss, FFT2D, 16, 1, 16),
]


def _uneven_grid_shard_sizes(n_points: int) -> list[int]:
    first = n_points // 2 + 3
    return [first, n_points - first]


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
            expected = loss(unsharded_pred, target, squash=False)
            expected.sum().backward()

            # Each rank's gradient should match the unsharded gradient on its own grid points.
            sharded_pred = pred[:, :, :, grid_shard_slice].clone().requires_grad_()
            actual = loss(
                sharded_pred,
                target[:, :, :, grid_shard_slice],
                squash=False,
                grid_shard_slice=grid_shard_slice,
                group=dist.group.WORLD,
                grid_dim=TensorDim.GRID,
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
