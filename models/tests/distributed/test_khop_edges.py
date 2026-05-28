# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for khop_edges: verifies fast path (dst-sorted) matches slow path."""

import pytest
import torch

from anemoi.models.distributed.khop_edges import _sort_edges_1hop_chunks_subgraph
from anemoi.models.distributed.khop_edges import build_graph_partition
from anemoi.models.distributed.khop_edges import sort_edge_index_by_dst
from anemoi.models.distributed.khop_edges import sort_edges_1hop_chunks


def _make_random_graph(num_src: int, num_dst: int, num_edges: int, seed: int = 42) -> tuple:
    """Create a random bipartite graph with random edge attributes."""
    gen = torch.Generator().manual_seed(seed)
    src = torch.randint(0, num_src, (num_edges,), generator=gen)
    dst = torch.randint(0, num_dst, (num_edges,), generator=gen)
    edge_index = torch.stack([src, dst])
    edge_attr = torch.randn(num_edges, 3, generator=gen)
    return edge_attr, edge_index


def _make_random_non_bipartite_graph(num_nodes: int, num_edges: int, seed: int = 42) -> tuple:
    """Create a random non-bipartite (self-loop) graph."""
    gen = torch.Generator().manual_seed(seed)
    src = torch.randint(0, num_nodes, (num_edges,), generator=gen)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=gen)
    edge_index = torch.stack([src, dst])
    edge_attr = torch.randn(num_edges, 3, generator=gen)
    return edge_attr, edge_index


def _edges_cover_same_set(
    attr_list_a: list[torch.Tensor],
    idx_list_a: list[torch.Tensor],
    attr_list_b: list[torch.Tensor],
    idx_list_b: list[torch.Tensor],
) -> None:
    """Assert two chunked edge lists cover the same set of edges per chunk (ignoring order within chunk)."""
    assert len(attr_list_a) == len(attr_list_b), "Different number of chunks"

    for i, (attr_a, idx_a, attr_b, idx_b) in enumerate(zip(attr_list_a, idx_list_a, attr_list_b, idx_list_b)):
        # Both chunks should have the same dst nodes
        dst_a = torch.unique(idx_a[1])
        dst_b = torch.unique(idx_b[1])
        assert torch.equal(dst_a.sort().values, dst_b.sort().values), (
            f"Chunk {i}: different dst nodes. "
            f"Subgraph: {dst_a.sort().values.tolist()}, Fast: {dst_b.sort().values.tolist()}"
        )

        # Same number of edges (same dst nodes → same edges in 1-hop)
        assert (
            attr_a.shape[0] == attr_b.shape[0]
        ), f"Chunk {i}: different number of edges ({attr_a.shape[0]} vs {attr_b.shape[0]})"

        # Edges should be the same set (sort by (src, dst) to compare)
        def _sort_edges(edge_index, edge_attr):
            key = edge_index[0] * 10_000_000 + edge_index[1]
            perm = key.argsort(stable=True)
            return edge_index[:, perm], edge_attr[perm]

        idx_a_s, attr_a_s = _sort_edges(idx_a, attr_a)
        idx_b_s, attr_b_s = _sort_edges(idx_b, attr_b)

        assert torch.equal(idx_a_s, idx_b_s), f"Chunk {i}: different edges"
        assert torch.allclose(attr_a_s, attr_b_s), f"Chunk {i}: different edge attributes"


class TestSortEdges1hopChunksBipartite:
    """Test fast vs slow path for bipartite graphs."""

    @pytest.mark.parametrize("num_chunks", [2, 3, 4, 7])
    @pytest.mark.parametrize(
        "num_src,num_dst,num_edges",
        [(10, 20, 50), (50, 30, 200), (100, 100, 500)],
    )
    def test_fast_matches_slow(self, num_chunks, num_src, num_dst, num_edges):
        """Starting from unsorted edges: sort + fast path should match slow path."""
        edge_attr, edge_index = _make_random_graph(num_src, num_dst, num_edges)
        num_nodes = (num_src, num_dst)

        # Slow path: works on unsorted edges
        attr_list_slow, idx_list_slow = _sort_edges_1hop_chunks_subgraph(num_nodes, edge_attr, edge_index, num_chunks)

        # Fast path: sort first, then use fast chunking
        edge_index_sorted, perm = sort_edge_index_by_dst(edge_index, max_value=num_dst)
        edge_attr_sorted = edge_attr[perm]

        attr_list_fast, idx_list_fast = sort_edges_1hop_chunks(
            num_nodes, edge_attr_sorted, edge_index_sorted, num_chunks, edges_are_dst_sorted=True
        )

        _edges_cover_same_set(attr_list_slow, idx_list_slow, attr_list_fast, idx_list_fast)

    @pytest.mark.parametrize("num_chunks", [2, 4])
    def test_fast_path_edge_splits_match_partition(self, num_chunks):
        """Edge splits from fast path should match GraphPartition metadata."""
        num_src, num_dst, num_edges = 30, 40, 150
        edge_attr, edge_index = _make_random_graph(num_src, num_dst, num_edges)
        num_nodes = (num_src, num_dst)

        edge_index_sorted, perm = sort_edge_index_by_dst(edge_index, max_value=num_dst)
        edge_attr_sorted = edge_attr[perm]

        partition = build_graph_partition(edge_index_sorted, num_chunks, num_nodes)
        attr_list, idx_list = sort_edges_1hop_chunks(
            num_nodes, edge_attr_sorted, edge_index_sorted, num_chunks, edges_are_dst_sorted=True
        )

        for i, (attr_chunk, idx_chunk) in enumerate(zip(attr_list, idx_list)):
            assert (
                attr_chunk.shape[0] == partition.edge_splits[i]
            ), f"Chunk {i}: attr size {attr_chunk.shape[0]} != partition split {partition.edge_splits[i]}"
            assert idx_chunk.shape[1] == partition.edge_splits[i]

    def test_total_edges_preserved(self):
        """Total edges across all chunks should equal original edge count."""
        num_src, num_dst, num_edges = 20, 30, 100
        num_chunks = 3
        edge_attr, edge_index = _make_random_graph(num_src, num_dst, num_edges)
        num_nodes = (num_src, num_dst)

        edge_index_sorted, perm = sort_edge_index_by_dst(edge_index, max_value=num_dst)
        edge_attr_sorted = edge_attr[perm]

        attr_list, idx_list = sort_edges_1hop_chunks(
            num_nodes, edge_attr_sorted, edge_index_sorted, num_chunks, edges_are_dst_sorted=True
        )

        total_edges = sum(a.shape[0] for a in attr_list)
        assert total_edges == num_edges


class TestSortEdges1hopChunksNonBipartite:
    """Test fast vs slow path for non-bipartite (self-loop) graphs."""

    @pytest.mark.parametrize("num_chunks", [2, 3, 5])
    @pytest.mark.parametrize("num_nodes,num_edges", [(20, 80), (50, 200), (100, 500)])
    def test_fast_matches_slow(self, num_chunks, num_nodes, num_edges):
        """Starting from unsorted edges: sort + fast path should match slow path."""
        edge_attr, edge_index = _make_random_non_bipartite_graph(num_nodes, num_edges)

        # Slow path: works on unsorted edges
        attr_list_slow, idx_list_slow = _sort_edges_1hop_chunks_subgraph(num_nodes, edge_attr, edge_index, num_chunks)

        # Fast path: sort first, then use fast chunking
        edge_index_sorted, perm = sort_edge_index_by_dst(edge_index, max_value=num_nodes)
        edge_attr_sorted = edge_attr[perm]

        attr_list_fast, idx_list_fast = sort_edges_1hop_chunks(
            num_nodes, edge_attr_sorted, edge_index_sorted, num_chunks, edges_are_dst_sorted=True
        )

        _edges_cover_same_set(attr_list_slow, idx_list_slow, attr_list_fast, idx_list_fast)

    def test_total_edges_preserved(self):
        """Total edges across all chunks should equal original edge count."""
        num_nodes, num_edges = 30, 120
        num_chunks = 4
        edge_attr, edge_index = _make_random_non_bipartite_graph(num_nodes, num_edges)

        edge_index_sorted, perm = sort_edge_index_by_dst(edge_index, max_value=num_nodes)
        edge_attr_sorted = edge_attr[perm]

        attr_list, idx_list = sort_edges_1hop_chunks(
            num_nodes, edge_attr_sorted, edge_index_sorted, num_chunks, edges_are_dst_sorted=True
        )

        total_edges = sum(a.shape[0] for a in attr_list)
        assert total_edges == num_edges


class TestSortEdges1hopChunksFallback:
    """Test that passing edges_are_dst_sorted=False uses the slow path."""

    def test_unsorted_edges_still_work(self):
        """sort_edges_1hop_chunks with edges_are_dst_sorted=False should work on unsorted edges."""
        num_src, num_dst, num_edges = 15, 25, 60
        num_chunks = 3
        edge_attr, edge_index = _make_random_graph(num_src, num_dst, num_edges)
        num_nodes = (num_src, num_dst)

        # This should not crash — uses slow path internally
        attr_list, idx_list = sort_edges_1hop_chunks(
            num_nodes, edge_attr, edge_index, num_chunks, edges_are_dst_sorted=False
        )

        total_edges = sum(a.shape[0] for a in attr_list)
        assert total_edges == num_edges


class TestBuildGraphPartition:
    """Test build_graph_partition correctness."""

    def test_edge_splits_sum_to_total(self):
        """Edge splits should sum to total number of edges."""
        num_src, num_dst, num_edges = 20, 30, 100
        num_parts = 4
        edge_attr, edge_index = _make_random_graph(num_src, num_dst, num_edges)

        edge_index_sorted, _ = sort_edge_index_by_dst(edge_index, max_value=num_dst)

        partition = build_graph_partition(edge_index_sorted, num_parts, (num_src, num_dst))
        assert sum(partition.edge_splits) == num_edges
        assert sum(partition.dst_splits) == num_dst

    def test_dst_splits_balanced(self):
        """Dst splits should be balanced (differ by at most 1)."""
        num_dst = 17
        num_parts = 4
        edge_attr, edge_index = _make_random_graph(10, num_dst, 50)
        edge_index_sorted, _ = sort_edge_index_by_dst(edge_index, max_value=num_dst)

        partition = build_graph_partition(edge_index_sorted, num_parts, (10, num_dst))
        sizes = partition.dst_splits
        assert max(sizes) - min(sizes) <= 1
