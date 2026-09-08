# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Distributed tests for residual connections that split the grid across ranks.

``SpectralOrnsteinConnection`` builds its fields from spherical harmonics, which
need the whole globe, so it has to move data around when the grid is split up.
These tests check that splitting the grid changes nothing: the outputs and the
gradients must match a single-rank run.

These tests are skipped by default. Pass ``--distributed`` to run them. Use
``--distributed-backend`` and ``--distributed-world-size`` to select the backend
and rank count.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.distributed as dist
from distributed_runner import run_distributed_test
from torch_geometric.data import HeteroData

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.models.layers.residual import SpectralOrnsteinConnection

NLAT = 8
ATOL = 1e-11
RTOL = 1e-11


def _regular_graph(nlat: int) -> HeteroData:
    """A regular lat-lon grid with ``nlat`` latitudes and twice as many longitudes."""
    lats = np.linspace(-90, 90, nlat)
    lons = np.linspace(0, 360, 2 * nlat, endpoint=False)
    graph = HeteroData()
    graph["data"].x = torch.tensor([(lat, lon) for lat in lats for lon in lons], dtype=torch.float32)
    return graph


def _data_indices(n_prognostic: int) -> MagicMock:
    data_indices = MagicMock()
    data_indices.model.input.prognostic = list(range(n_prognostic))
    data_indices.model.input.name_to_index = {f"var{i}": i for i in range(n_prognostic)}
    data_indices.data.input.prognostic = list(range(n_prognostic))
    return data_indices


def _build_connection(
    n_prognostic: int,
    truncate: bool,
    anti_aliasing: bool,
    device: torch.device,
) -> SpectralOrnsteinConnection:
    """Build the layer with the same parameter values on every rank."""
    connection = SpectralOrnsteinConnection(
        lmax=2,
        grid="regular",
        truncate=truncate,
        anti_aliasing=anti_aliasing,
        graph=_regular_graph(NLAT),
        data_indices=_data_indices(n_prognostic),
        dataset_name="data",
    )
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in connection.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator))
    return connection.to(device=device, dtype=torch.float64)


def _test_sharded_matches_unsharded(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    truncate: bool,
    anti_aliasing: bool,
) -> None:
    """Run the layer on the full grid and on a split grid, then compare both.

    The gradients of a split run are partial: each rank only sees its own points
    and its own variables. Adding them up across the group must give back the
    single-rank gradient, which is what the training strategy relies on when it
    scales gradients by the size of the model communication group.
    """
    n_prognostic = max(4, world_size)  # the variables get split up too, so every rank needs one
    reference = _build_connection(n_prognostic, truncate, anti_aliasing, device)
    sharded = _build_connection(n_prognostic, truncate, anti_aliasing, device)

    n_points = NLAT * 2 * NLAT
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(2, 3, 1, n_points, n_prognostic, generator=generator, dtype=torch.float64).to(device)
    loss_weights = torch.randn(2, 1, n_points, n_prognostic, generator=generator, dtype=torch.float64).to(device)

    x_before = x.clone()
    out_reference = reference(x)
    (out_reference * loss_weights).sum().backward()
    torch.testing.assert_close(x, x_before, atol=0.0, rtol=0.0, msg="The layer must not modify its input.")

    grid_shard_sizes = get_balanced_partition_sizes(n_points, world_size)
    start, end = get_partition_range(grid_shard_sizes, rank)
    out_sharded = sharded(x[..., start:end, :], grid_shard_sizes=grid_shard_sizes, model_comm_group=group)
    (out_sharded * loss_weights[..., start:end, :]).sum().backward()

    torch.testing.assert_close(out_sharded, out_reference[..., start:end, :], atol=ATOL, rtol=RTOL)

    for (name, expected), (_, actual) in zip(reference.named_parameters(), sharded.named_parameters()):
        # Every parameter has to receive a gradient, otherwise DDP stalls on the next step.
        assert expected.grad is not None, f"{name!r} received no gradient on the full grid."
        assert actual.grad is not None, f"{name!r} received no gradient on rank {rank}."

        summed = actual.grad.clone()
        dist.all_reduce(summed, op=dist.ReduceOp.SUM, group=group)
        torch.testing.assert_close(
            summed,
            expected.grad,
            atol=ATOL,
            rtol=RTOL,
            msg=f"Gradients of {name!r} do not add up to the unsharded gradient.",
        )


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("truncate", "anti_aliasing"),
    [
        pytest.param(False, False, id="no_truncation"),
        pytest.param(True, False, id="truncation"),
        pytest.param(True, True, id="truncation_anti_aliasing"),
    ],
)
def test_spectral_ornstein_sharded_matches_unsharded(
    truncate: bool, anti_aliasing: bool, distributed_backend: str, distributed_world_size: int
) -> None:
    run_distributed_test(
        _test_sharded_matches_unsharded,
        backend=distributed_backend,
        world_size=distributed_world_size,
        truncate=truncate,
        anti_aliasing=anti_aliasing,
    )


def _test_too_few_variables_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
) -> None:
    """One variable cannot be spread over two or more ranks, and saying so beats an FFT crash."""
    connection = _build_connection(n_prognostic=1, truncate=True, anti_aliasing=True, device=device)

    n_points = NLAT * 2 * NLAT
    grid_shard_sizes = get_balanced_partition_sizes(n_points, world_size)
    start, end = get_partition_range(grid_shard_sizes, rank)
    x = torch.zeros(1, 2, 1, end - start, 1, dtype=torch.float64, device=device)

    with pytest.raises(AssertionError, match="truncated variables"):
        connection(x, grid_shard_sizes=grid_shard_sizes, model_comm_group=group)


@pytest.mark.distributed
def test_spectral_ornstein_rejects_more_ranks_than_truncated_variables(
    distributed_backend: str, distributed_world_size: int
) -> None:
    run_distributed_test(
        _test_too_few_variables_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
    )
