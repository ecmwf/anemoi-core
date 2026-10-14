# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from torch_geometric.utils import degree

from anemoi.models.distributed.khop_edges import build_chunk_plan
from anemoi.models.distributed.khop_edges import build_graph_partition
from anemoi.models.distributed.khop_edges import sort_edge_index_by_dst


def _random_dst_sorted_graph(num_src: int, num_dst: int, num_edges: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    edge_index = torch.stack(
        [
            torch.randint(0, num_src, (num_edges,), generator=generator),
            torch.randint(0, num_dst, (num_edges,), generator=generator),
        ]
    )
    return sort_edge_index_by_dst(edge_index, max_value=num_dst)[0]


@pytest.mark.parametrize("num_parts", [1, 2, 3, 7, 16])
@pytest.mark.parametrize("num_dst", [5, 40])
def test_edge_splits_match_degree(num_parts: int, num_dst: int) -> None:
    edge_index = _random_dst_sorted_graph(30, num_dst, 200)
    partition = build_graph_partition(edge_index, num_parts, (30, num_dst))
    degree_per_dst = degree(edge_index[1], num_nodes=num_dst, dtype=torch.long)
    expected = [int(chunk.sum()) for chunk in torch.split(degree_per_dst, partition.dst_splits)]
    assert partition.edge_splits == expected


def test_edge_splits_with_isolated_dst_nodes() -> None:
    # dst nodes 0-3 and 7-9 have no incoming edges
    edge_index = torch.tensor([[0, 1, 2, 3], [4, 5, 5, 6]])
    partition = build_graph_partition(edge_index, 4, (4, 10))
    assert sum(partition.edge_splits) == 4
    degree_per_dst = degree(edge_index[1], num_nodes=10, dtype=torch.long)
    assert partition.edge_splits == [int(c.sum()) for c in torch.split(degree_per_dst, partition.dst_splits)]


@pytest.mark.parametrize("num_parts", [1, 3, 8])
def test_chunk_plan_matches_partition_materialise(num_parts: int) -> None:
    num_src, num_dst = 50, 24
    edge_index = _random_dst_sorted_graph(num_src, num_dst, 300, seed=1)
    x = (torch.randn(num_src, 3), torch.randn(num_dst, 3))
    edge_attr = torch.randn(edge_index.size(1), 2)
    cond = (torch.randn(num_src, 1), torch.randn(num_dst, 1))

    partition = build_graph_partition(edge_index, num_parts, (num_src, num_dst))
    plan = build_chunk_plan(edge_index, num_parts, (num_src, num_dst))
    assert plan.num_parts == num_parts
    for chunk_id in range(num_parts):
        expected = partition.materialise(chunk_id, x, edge_attr, edge_index, cond=cond)
        actual = plan.materialise(chunk_id, x, edge_attr, cond=cond)
        torch.testing.assert_close(actual[0][0], expected[0][0])
        torch.testing.assert_close(actual[0][1], expected[0][1])
        torch.testing.assert_close(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2])
        torch.testing.assert_close(actual[3][0], expected[3][0])
        torch.testing.assert_close(actual[3][1], expected[3][1])
