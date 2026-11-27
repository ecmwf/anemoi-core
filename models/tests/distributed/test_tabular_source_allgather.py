# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Distributed tests for :meth:`TabularSource.allgather`.

Each rank holds its part of every time window, split as the tabular reader splits it at read
time (a balanced partition per window). Gathering over the group must give every rank the full,
unsharded source back, window by window.

These tests are skipped by default. Pass ``--distributed`` to run them. The single-rank paths
are covered by ``tests/data/test_source.py::TestAllgather``.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from anemoi.models.data import TensorLayout
from anemoi.models.data.sources import TabularSource
from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.balanced_partition import get_partition_range

from ._distributed_runner import _run_distributed_test

N_VARIABLES = 3
WINDOW_SPACING = 3600.0


def _full_source(window_sizes: list[list[int]], n_members: int | None, device: torch.device) -> TabularSource:
    """Build the unsharded source: distinct values, coordinates and timedeltas for every point."""
    data, coordinates, timedeltas, boundaries = [], [], [], []
    for sample_idx, sizes in enumerate(window_sizes):
        n_points = sum(sizes)
        values = torch.arange(n_points * N_VARIABLES, dtype=torch.float32, device=device).reshape(n_points, N_VARIABLES)
        values = values + 1000.0 * sample_idx
        if n_members is not None:
            values = torch.stack([values + 0.1 * member for member in range(n_members)])
        data.append(values)
        index = torch.arange(n_points, dtype=torch.float32, device=device)
        coordinates.append(torch.stack([index, torch.full_like(index, float(sample_idx))], dim=1))
        window_index = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes, dtype=torch.long))
        timedeltas.append(WINDOW_SPACING * window_index.float().to(device) + index)
        starts = [sum(sizes[:w]) for w in range(len(sizes))]
        boundaries.append(tuple(slice(start, start + size) for start, size in zip(starts, sizes)))
    return TabularSource(
        name="obs",
        variables=[f"v{i}" for i in range(N_VARIABLES)],
        layout=TensorLayout(grid=0, variables=1)
        if n_members is None
        else TensorLayout(ensemble=0, grid=1, variables=2),
        values=data,
        coordinates=coordinates,
        timedeltas=timedeltas,
        boundaries=boundaries,
    )


def _rank_view(full: TabularSource, rank: int, world_size: int) -> TabularSource:
    """Return this rank's part of every window, as the tabular reader shards at read time."""
    grid_axis = full.layout.grid
    data, coordinates, timedeltas, boundaries, shard_sizes = [], [], [], [], []
    for sample_idx, sample_boundaries in enumerate(full.boundaries):
        local_slices, local_boundaries, window_sizes = [], [], []
        offset = 0
        for window in sample_boundaries:
            sizes = get_balanced_partition_sizes(window.stop - window.start, world_size)
            start, end = get_partition_range(sizes, rank)
            local_slices.append(slice(window.start + start, window.start + end))
            local_boundaries.append(slice(offset, offset + end - start))
            window_sizes.append(sizes)
            offset += end - start
        index = torch.cat([torch.arange(s.start, s.stop) for s in local_slices]).to(full.device)
        data.append(full.values[sample_idx].index_select(grid_axis, index))
        coordinates.append(full.coordinates[sample_idx][index])
        timedeltas.append(full.timedeltas[sample_idx][index])
        boundaries.append(tuple(local_boundaries))
        shard_sizes.append(window_sizes)
    return full.clone(
        values=data, coordinates=coordinates, timedeltas=timedeltas, boundaries=boundaries, shard_sizes=shard_sizes
    )


def _check_tabular_allgather(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    window_sizes: list[list[int]],
    n_members: int | None,
) -> None:
    full = _full_source(window_sizes, n_members, device)
    local = _rank_view(full, rank, world_size)
    assert local.is_sharded

    gathered = local.allgather(group)

    assert gathered.shard_sizes is None
    assert not gathered.is_sharded
    # allgather returns each sample's windows as a list, select_time and concat_time as a tuple.
    assert [tuple(windows) for windows in gathered.boundaries] == list(full.boundaries)
    for field_name in ("values", "coordinates", "timedeltas"):
        for sample_idx, (got, want) in enumerate(zip(getattr(gathered, field_name), getattr(full, field_name))):
            assert got.device == want.device
            assert torch.equal(got, want), f"rank {rank}: {field_name} of sample {sample_idx} differs"
    # A gathered source is replicated: gathering again is a no-op, as the plot callbacks rely on.
    assert gathered.allgather(group) is gathered


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("window_sizes", "n_members"),
    [
        pytest.param([[5, 3]], None, id="one-sample-two-windows"),
        pytest.param([[7, 1, 4], [2, 6, 3]], None, id="ragged-samples"),
        # Windows smaller than the group leave some ranks with no points in them.
        pytest.param([[1, 0, 2], [0, 0, 0]], None, id="empty-windows-and-ranks"),
        pytest.param([[4, 3], [5, 1]], 2, id="ensemble-axis"),
    ],
)
def test_tabular_allgather_restores_every_window(
    window_sizes: list[list[int]],
    n_members: int | None,
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    _run_distributed_test(
        _check_tabular_allgather,
        backend=distributed_backend,
        world_size=distributed_world_size,
        window_sizes=window_sizes,
        n_members=n_members,
    )
