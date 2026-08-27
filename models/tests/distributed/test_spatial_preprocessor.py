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
