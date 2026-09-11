# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
from distributed_runner import run_distributed_test
from torch_geometric.data import HeteroData

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.preprocessing.cross_grid_projector import CrossGridProjector


def _test_cross_grid_projector_returns_target_shards_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
) -> None:
    source_grid_size = world_size * 4
    target_grid_size = world_size * 2
    variables = world_size

    graph = HeteroData()
    graph["source"].num_nodes = source_grid_size
    graph["target"].num_nodes = target_grid_size
    graph["source", "to", "target"].edge_index = torch.stack(
        (
            torch.arange(source_grid_size),
            torch.arange(source_grid_size) // 2,
        )
    )
    projector = CrossGridProjector(
        graph=graph,
        edges_name=("source", "to", "target"),
        row_normalize=False,
    )

    full = torch.arange(source_grid_size * variables, dtype=torch.float32, device=device).reshape(
        1,
        1,
        1,
        source_grid_size,
        variables,
    )
    source_grid_shard_sizes = get_balanced_partition_sizes(source_grid_size, world_size)
    target_grid_shard_sizes = get_balanced_partition_sizes(target_grid_size, world_size)
    local = torch.split(full, source_grid_shard_sizes, dim=-2)[rank].contiguous()

    projected, returned_grid_shard_sizes = projector(
        local,
        model_comm_group=group,
        grid_shard_sizes=source_grid_shard_sizes,
    )

    expected_full = full.reshape(1, 1, 1, target_grid_size, 2, variables).sum(dim=-2)
    expected_local = torch.split(expected_full, target_grid_shard_sizes, dim=-2)[rank].contiguous()
    assert returned_grid_shard_sizes == target_grid_shard_sizes
    torch.testing.assert_close(projected, expected_local)


@pytest.mark.distributed
def test_cross_grid_projector_returns_target_shards(
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    run_distributed_test(
        _test_cross_grid_projector_returns_target_shards_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
    )


def _test_cross_grid_projector_fewer_vars_than_ranks_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
) -> None:
    """Projection with fewer variables than ranks uses the gather-project-shard fallback.

    Exercises the path where ``min(channel_shard_sizes) == 0``.
    """
    source_grid_size = world_size * 4
    target_grid_size = world_size * 2
    # Fewer variables than ranks triggers the fallback path.
    variables = 1

    graph = HeteroData()
    graph["source"].num_nodes = source_grid_size
    graph["target"].num_nodes = target_grid_size
    graph["source", "to", "target"].edge_index = torch.stack(
        (
            torch.arange(source_grid_size),
            torch.arange(source_grid_size) // 2,
        )
    )
    projector = CrossGridProjector(
        graph=graph,
        edges_name=("source", "to", "target"),
        row_normalize=False,
    )

    full = torch.arange(source_grid_size * variables, dtype=torch.float32, device=device).reshape(
        1,
        1,
        1,
        source_grid_size,
        variables,
    )
    source_grid_shard_sizes = get_balanced_partition_sizes(source_grid_size, world_size)
    target_grid_shard_sizes = get_balanced_partition_sizes(target_grid_size, world_size)
    local = torch.split(full, source_grid_shard_sizes, dim=-2)[rank].contiguous()

    projected, returned_grid_shard_sizes = projector(
        local,
        model_comm_group=group,
        grid_shard_sizes=source_grid_shard_sizes,
    )

    # Each target node sums 2 source nodes (ratio = source_grid_size // target_grid_size = 2).
    expected_full = full.reshape(1, 1, 1, target_grid_size, 2, variables).sum(dim=-2)
    expected_local = torch.split(expected_full, target_grid_shard_sizes, dim=-2)[rank].contiguous()
    assert returned_grid_shard_sizes == target_grid_shard_sizes
    torch.testing.assert_close(projected, expected_local)


@pytest.mark.distributed
def test_cross_grid_projector_fewer_vars_than_ranks(
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    run_distributed_test(
        _test_cross_grid_projector_fewer_vars_than_ranks_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
    )


_GRADIENT_TOLERANCE = 64 * torch.finfo(torch.float64).eps


def _build_overlapping_projector(source_grid_size: int, target_grid_size: int) -> CrossGridProjector:
    """Build a projector whose target nodes have edges across shard boundaries."""

    ratio = source_grid_size // target_grid_size
    source_index = torch.arange(source_grid_size)
    target_index = source_index // ratio

    edge_index = torch.stack(
        (
            source_index.repeat(2),
            torch.cat((target_index, (target_index + 1) % target_grid_size)),
        )
    )
    # Deterministic, strictly positive, non-uniform weights: identical on every rank.
    edge_attr = 1.0 + torch.sin(torch.arange(edge_index.shape[1], dtype=torch.float32)) ** 2

    graph = HeteroData()
    graph["source"].num_nodes = source_grid_size
    graph["target"].num_nodes = target_grid_size
    graph["source", "to", "target"].edge_index = edge_index
    graph["source", "to", "target"].edge_attr = edge_attr

    return CrossGridProjector(
        graph=graph,
        edges_name=("source", "to", "target"),
        edge_weight_attribute="edge_attr",
        row_normalize=True,
    )


def _test_cross_grid_projector_gradients_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    variables: int,
    expect_transpose_path: bool,
) -> None:
    """Compare sharded backward passes against the replicated reference in float64.

    ``variables >= world_size`` exercises the all-to-all transpose path,
    ``variables < world_size`` the gather-project-shard fallback.
    """
    source_grid_size = world_size * 6
    target_grid_size = world_size * 3
    batch, time, ensemble = 2, 3, 2

    channel_shard_sizes = get_balanced_partition_sizes(variables, world_size)
    assert (min(channel_shard_sizes) > 0) is expect_transpose_path

    projector = _build_overlapping_projector(source_grid_size, target_grid_size)

    generator = torch.Generator().manual_seed(0)
    full = torch.randn(batch, time, ensemble, source_grid_size, variables, generator=generator, dtype=torch.float64).to(
        device
    )
    grad_weights = torch.randn(
        batch, time, ensemble, target_grid_size, variables, generator=generator, dtype=torch.float64
    ).to(device)

    reference_input = full.clone().requires_grad_(True)
    reference_output, reference_shard_sizes = projector(reference_input, model_comm_group=None, grid_shard_sizes=None)
    assert reference_shard_sizes is None
    (reference_output * grad_weights).sum().backward()

    source_grid_shard_sizes = get_balanced_partition_sizes(source_grid_size, world_size)
    target_grid_shard_sizes = get_balanced_partition_sizes(target_grid_size, world_size)

    local_input = torch.split(full, source_grid_shard_sizes, dim=-2)[rank].contiguous().requires_grad_(True)
    local_output, returned_grid_shard_sizes = projector(
        local_input,
        model_comm_group=group,
        grid_shard_sizes=source_grid_shard_sizes,
    )
    assert returned_grid_shard_sizes == target_grid_shard_sizes

    # Summing the per-rank losses reproduces the reference loss, so each rank's
    # local gradient must equal the matching slice of the reference gradient.
    local_grad_weights = torch.split(grad_weights, target_grid_shard_sizes, dim=-2)[rank]
    (local_output * local_grad_weights).sum().backward()

    expected_output = torch.split(reference_output.detach(), target_grid_shard_sizes, dim=-2)[rank]
    expected_grad = torch.split(reference_input.grad, source_grid_shard_sizes, dim=-2)[rank]

    torch.testing.assert_close(
        local_output.detach(), expected_output, rtol=_GRADIENT_TOLERANCE, atol=_GRADIENT_TOLERANCE
    )
    torch.testing.assert_close(local_input.grad, expected_grad, rtol=_GRADIENT_TOLERANCE, atol=_GRADIENT_TOLERANCE)


@pytest.mark.distributed
@pytest.mark.parametrize("expect_transpose_path", [True, False])
def test_cross_grid_projector_gradients_match_replicated(
    expect_transpose_path: bool,
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    run_distributed_test(
        _test_cross_grid_projector_gradients_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        variables=distributed_world_size + 1 if expect_transpose_path else 1,
        expect_transpose_path=expect_transpose_path,
    )
