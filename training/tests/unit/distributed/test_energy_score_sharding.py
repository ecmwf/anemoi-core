# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Compare grid-sharded energy scores and their gradients with the unsharded ones on CPU."""

from datetime import timedelta
from itertools import combinations
from pathlib import Path
from typing import get_args

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from anemoi.training.losses import EnergyScoreLoss
from anemoi.training.utils.enums import EnergyScoreDimension, TensorDim

GRID_SHARD_SIZES = [3, 2]


def _check_sharded_energy_scores(rank: int, init_file: str) -> None:
    torch.set_num_threads(1)
    world_size = len(GRID_SHARD_SIZES)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        # Every rank builds the same full inputs and keeps its own block of grid nodes.
        generator = torch.Generator().manual_seed(0)
        pred = torch.randn(2, 3, 4, sum(GRID_SHARD_SIZES), 2, dtype=torch.float64, generator=generator)
        target = torch.randn(2, 3, 1, sum(GRID_SHARD_SIZES), 2, dtype=torch.float64, generator=generator)
        start = sum(GRID_SHARD_SIZES[:rank])
        grid_shard_slice = slice(start, start + GRID_SHARD_SIZES[rank])
        # With ignore_nans, one value is missing from the last rank's block of grid nodes.
        pred_with_gap = pred.clone()
        pred_with_gap[0, 0, 1, -1, 1] = torch.nan

        dimensions = get_args(EnergyScoreDimension)
        for ignore_nans, full_pred in [(False, pred), (True, pred_with_gap)]:
            for size in range(1, len(dimensions) + 1):
                for norm_over in combinations(dimensions, size):
                    loss = EnergyScoreLoss(norm_over=norm_over, ignore_nans=ignore_nans)
                    loss.add_scaler(TensorDim.TIME, torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64), name="time")
                    loss.add_scaler(
                        TensorDim.GRID,
                        torch.tensor([0.1, 0.3, 0.2, 0.25, 0.15], dtype=torch.float64),
                        name="grid",
                    )

                    unsharded_pred = full_pred.clone().requires_grad_()
                    expected = loss(unsharded_pred, target, squash=False)
                    expected.sum().backward()

                    # Each rank's gradient covers its own grid nodes and should equal that
                    # part of the unsharded gradient.
                    sharded_pred = full_pred[:, :, :, grid_shard_slice].clone().requires_grad_()
                    actual = loss(
                        sharded_pred,
                        target[:, :, :, grid_shard_slice],
                        squash=False,
                        grid_shard_slice=grid_shard_slice,
                        group=dist.group.WORLD,
                        grid_dim=TensorDim.GRID,
                        grid_shard_sizes=GRID_SHARD_SIZES,
                    )
                    actual.sum().backward()

                    msg = f"rank={rank}, norm_over={norm_over}, ignore_nans={ignore_nans}"
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
def test_sharded_energy_score_and_gradients_match_unsharded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    mp.spawn(
        _check_sharded_energy_scores,
        args=(str(tmp_path / "rendezvous"),),
        nprocs=len(GRID_SHARD_SIZES),
        join=True,
    )
