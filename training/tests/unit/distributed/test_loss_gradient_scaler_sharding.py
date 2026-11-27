# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Compare the loss-gradient rescaling on sharded predictions with a single-process reference on CPU."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from anemoi.models.data import GriddedSource
from anemoi.models.data import TabularSource
from anemoi.models.data import TensorLayout
from anemoi.training.losses import LossScaler

GRIDDED_SHAPE = (2, 2, 3, 13, 4)  # (batch, time, ensemble, grid, variable); 13 points split unevenly
TABULAR_SIZES = (2, 7, 11)  # the first sample has fewer points than some world sizes have ranks
N_VARIABLES = 4
GRIDDED_LAYOUT = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
VARIABLE_WEIGHTS = torch.tensor([1.0, 10.0, 100.0, 0.1])


def _upstream(seed: int, shape: tuple[int, ...]) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=generator) * VARIABLE_WEIGHTS


def _gridded_grad(upstream: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
    x = torch.zeros_like(upstream, requires_grad=True)
    source = GriddedSource(
        name="data",
        values=x,
        variables=[f"v{i}" for i in range(N_VARIABLES)],
        coordinates=torch.zeros(x.shape[GRIDDED_LAYOUT.grid], 2),
        layout=GRIDDED_LAYOUT,
    )
    (LossScaler()(source, group=group).values * upstream).sum().backward()
    return x.grad


def _tabular_grads(upstreams: list[torch.Tensor], group: dist.ProcessGroup | None) -> list[torch.Tensor]:
    xs = [torch.zeros_like(u, requires_grad=True) for u in upstreams]
    source = TabularSource(
        name="obs",
        values=xs,
        variables=[f"v{i}" for i in range(N_VARIABLES)],
        coordinates=[torch.zeros(u.shape[0], 2) for u in upstreams],
        layout=TensorLayout(grid=0, variables=1),
        timedeltas=[torch.zeros(u.shape[0]) for u in upstreams],
        boundaries=[(slice(0, u.shape[0]),) for u in upstreams],
    )
    out = LossScaler()(source, group=group)
    sum((v * u).sum() for v, u in zip(out.values, upstreams, strict=True)).backward()
    return [x.grad for x in xs]


def _check_sharded_scaling(rank: int, world_size: int, init_file: str) -> None:
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

        # Gridded, grid sharded: the local gradient is the shard of the single-process gradient.
        upstream = _upstream(0, GRIDDED_SHAPE)
        expected = _gridded_grad(upstream, None)
        local = torch.tensor_split(torch.arange(GRIDDED_SHAPE[3]), world_size)[rank]
        actual = _gridded_grad(upstream[:, :, :, local], group)
        if not torch.allclose(actual, expected[:, :, :, local], rtol=1e-10, atol=1e-12):
            mismatches.append("gridded, grid sharded: gradient differs")

        # Gridded, replicated but reduced over the group anyway: the result does not change.
        if not torch.allclose(_gridded_grad(upstream, group), expected, rtol=1e-10, atol=1e-12):
            mismatches.append("gridded, replicated: gradient differs")

        # Tabular, every sample's points split over the ranks (one window per sample).
        upstreams = [_upstream(seed, (size, N_VARIABLES)) for seed, size in enumerate(TABULAR_SIZES, start=1)]
        expected_tabular = _tabular_grads(upstreams, None)
        locals_ = [torch.tensor_split(torch.arange(size), world_size)[rank] for size in TABULAR_SIZES]
        actual_tabular = _tabular_grads([u[idx] for u, idx in zip(upstreams, locals_, strict=True)], group)
        for i, (got, ref, idx) in enumerate(zip(actual_tabular, expected_tabular, locals_, strict=True)):
            if not torch.allclose(got, ref[idx], rtol=1e-10, atol=1e-12):
                mismatches.append(f"tabular sample {i} ({TABULAR_SIZES[i]} points): gradient differs")

        assert not mismatches, f"rank {rank} of {world_size}:\n" + "\n".join(mismatches)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Requires Gloo")
@pytest.mark.parametrize("world_size", [2, 3])
def test_sharded_loss_gradient_scaling_matches_single_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
) -> None:
    """Rescaling a prediction split over ranks gives each rank its shard of the single-process gradient."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    mp.spawn(_check_sharded_scaling, args=(world_size, str(tmp_path / "rendezvous")), nprocs=world_size, join=True)
