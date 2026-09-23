# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Halo metadata unit tests with mocked process groups and communication."""

from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

from anemoi.models.distributed import halo
from anemoi.models.distributed.khop_edges import GraphPartition

# (src_splits, dst_splits, edges as (src, dst) pairs with global node IDs)
GRAPH_CASES = {
    "symmetric": (
        [2, 3, 1],
        [2, 3, 1],
        [(0, 2), (2, 0), (1, 5), (5, 1), (4, 5), (5, 4), (2, 2)],
    ),
    "directed": (
        [2, 3, 1],
        [2, 3, 1],
        [(1, 2), (0, 2), (0, 2), (0, 4), (1, 5), (4, 5), (2, 2)],
    ),
    "bipartite": (
        [2, 3, 1],
        [3, 1, 4],
        [(5, 0), (2, 0), (2, 0), (4, 1), (0, 3), (1, 4), (4, 6), (3, 7), (5, 7)],
    ),
    "bipartite-equal-sizes": (
        [2, 3, 1],
        [2, 3, 1],
        [(0, 2), (1, 4), (3, 5), (2, 2)],
    ),
    "empty-partitions": (
        [0, 3, 2],
        [2, 0, 3],
        [(4, 0), (3, 0), (0, 1), (2, 2), (4, 3), (1, 4)],
    ),
    "local-only": ([2, 3, 1], [2, 3, 1], [(0, 1), (1, 0), (2, 4), (5, 5)]),
    "empty-edges": ([2, 3, 1], [2, 3, 1], []),
    "single-rank": ([3], [3], [(0, 2), (1, 1)]),
    "single-rank-bipartite": ([2], [3], [(0, 2), (1, 1)]),
}


def _make_graph(case: str) -> tuple[GraphPartition, torch.Tensor]:
    src_splits, dst_splits, edges = GRAPH_CASES[case]
    edge_index = torch.tensor(sorted(edges, key=lambda edge: edge[1]), dtype=torch.long)
    edge_index = edge_index.reshape(-1, 2).T.contiguous()
    dst_owners = _owners(dst_splits)
    edge_splits = [sum(dst_owners[dst] == rank for _, dst in edges) for rank in range(len(dst_splits))]
    partition = GraphPartition(
        num_nodes=(sum(src_splits), sum(dst_splits)),
        num_edges=len(edges),
        num_parts=len(dst_splits),
        dst_splits=dst_splits,
        edge_splits=edge_splits,
        src_splits=src_splits,
    )
    return partition, edge_index


def _owners(splits: list[int]) -> list[int]:
    return [rank for rank, size in enumerate(splits) for _ in range(size)]


def _remote_nodes(partition: GraphPartition, edge_index: torch.Tensor, sender: int, receiver: int) -> list[int]:
    """Sorted global IDs of the sender's source nodes that have an edge to one of the receiver's destination nodes."""
    if sender == receiver:
        return []
    src_owners, dst_owners = _owners(partition.src_splits), _owners(partition.dst_splits)
    return sorted(
        {src for src, dst in edge_index.T.tolist() if src_owners[src] == sender and dst_owners[dst] == receiver}
    )


def _fake_request_send_nodes(partition: GraphPartition, edge_index: torch.Tensor, rank: int, requests: list) -> Mock:
    """Stand-in for the all-to-all: records this rank's requests and answers with what the peers would ask for."""

    def request(recv_nodes_by_rank, model_comm_group):
        requests.append(recv_nodes_by_rank)
        return tuple(
            edge_index.new_tensor(_remote_nodes(partition, edge_index, rank, peer))
            for peer in range(partition.num_parts)
        )

    return Mock(side_effect=request)


def _assert_halo_info(info: halo.HaloInfo, partition: GraphPartition, edge_index: torch.Tensor, rank: int) -> None:
    """Check exact metadata against a reference built from node ownership."""
    src_owners, dst_owners = _owners(partition.src_splits), _owners(partition.dst_splits)
    local_src_ids = [node for node, owner in enumerate(src_owners) if owner == rank]
    local_dst_ids = [node for node, owner in enumerate(dst_owners) if owner == rank]
    expected_recv = [_remote_nodes(partition, edge_index, peer, rank) for peer in range(partition.num_parts)]
    expected_send = [_remote_nodes(partition, edge_index, rank, peer) for peer in range(partition.num_parts)]
    halo_ids = [node for peer_nodes in expected_recv for node in peer_nodes]
    src_to_local = {node: index for index, node in enumerate(local_src_ids + halo_ids)}
    dst_to_local = {node: index for index, node in enumerate(local_dst_ids)}
    expected_edges = [
        (src_to_local[src], dst_to_local[dst]) for src, dst in edge_index.T.tolist() if dst_owners[dst] == rank
    ]
    expected_edge_index = edge_index.new_tensor(expected_edges).reshape(-1, 2).T

    assert info.num_local_src_nodes == len(local_src_ids)
    assert info.num_local_dst_nodes == len(local_dst_ids)
    assert info.num_halo_nodes == len(halo_ids)
    assert info.total_src_nodes == len(local_src_ids) + len(halo_ids)
    assert info.recv_counts == tuple(map(len, expected_recv))
    assert info.send_counts == tuple(map(len, expected_send))
    assert len(info.send_indices) == partition.num_parts
    for actual, nodes in zip(info.send_indices, expected_send):
        torch.testing.assert_close(actual, edge_index.new_tensor([src_to_local[node] for node in nodes]))
    torch.testing.assert_close(info.edge_index_local, expected_edge_index)


def _mock_group(monkeypatch: pytest.MonkeyPatch, size: int, rank: int) -> Mock:
    group = Mock(spec=dist.ProcessGroup)
    group.size.return_value = size
    monkeypatch.setattr(dist, "get_rank", Mock(return_value=rank))
    return group


@pytest.mark.parametrize("case", GRAPH_CASES)
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("debug", [False, True])
def test_build_halo_info(case: str, sharded: bool, debug: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    partition, edge_index = _make_graph(case)
    original_edges = edge_index.clone()
    for rank in range(partition.num_parts):
        group = _mock_group(monkeypatch, partition.num_parts, rank)
        local_edges = edge_index.split(partition.edge_splits, dim=1)[rank].contiguous()
        original_local_edges = local_edges.clone()
        shard = Mock(return_value=local_edges)
        requests = []
        monkeypatch.setattr(halo, "shard_tensor", shard)
        monkeypatch.setattr(
            halo, "_request_send_nodes", _fake_request_send_nodes(partition, edge_index, rank, requests)
        )
        info = halo.build_halo_info(
            partition,
            local_edges if sharded else edge_index,
            group,
            edge_shard_sizes=partition.edge_splits if sharded else None,
            debug=debug,
        )
        _assert_halo_info(info, partition, edge_index, rank)
        (sent_requests,) = requests
        for peer, actual in enumerate(sent_requests):
            torch.testing.assert_close(actual, edge_index.new_tensor(_remote_nodes(partition, edge_index, peer, rank)))
        torch.testing.assert_close(edge_index, original_edges)
        torch.testing.assert_close(local_edges, original_local_edges)
        if sharded:
            shard.assert_not_called()
        else:
            shard.assert_called_once_with(edge_index, 1, partition.edge_splits, group)


def test_partition_must_match_group_size(monkeypatch: pytest.MonkeyPatch) -> None:
    partition, edge_index = _make_graph("directed")
    group = _mock_group(monkeypatch, partition.num_parts + 1, 0)
    with pytest.raises(AssertionError, match="Partition num_parts"):
        halo.build_halo_info(partition, edge_index, group)


def test_requires_sharded_source_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    partition, edge_index = _make_graph("directed")
    group = _mock_group(monkeypatch, partition.num_parts, 0)
    with pytest.raises(AssertionError, match="sharded source nodes"):
        halo.build_halo_info(replace(partition, src_splits=None), edge_index, group)


def test_debug_rejects_edges_to_other_ranks_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    partition, edge_index = _make_graph("directed")
    group = _mock_group(monkeypatch, partition.num_parts, 0)
    request = _fake_request_send_nodes(partition, edge_index, 0, [])
    monkeypatch.setattr(halo, "_request_send_nodes", request)
    # Rank 0 owns destinations 0 and 1; node 2 belongs to rank 1.
    foreign_edges = edge_index.new_tensor([[0], [2]])
    with pytest.raises(AssertionError, match="destination nodes"):
        halo.build_halo_info(partition, foreign_edges, group, edge_shard_sizes=[1, 0, 0], debug=True)


def test_debug_rejects_requests_for_nodes_owned_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    partition, edge_index = _make_graph("directed")
    group = _mock_group(monkeypatch, partition.num_parts, 0)
    # Rank 0 owns sources 0 and 1; rank 1 asks it for node 3.
    requested = (edge_index.new_tensor([]), edge_index.new_tensor([3]), edge_index.new_tensor([]))
    monkeypatch.setattr(halo, "_request_send_nodes", Mock(return_value=requested))
    local_edges = edge_index.split(partition.edge_splits, dim=1)[0]
    with pytest.raises(AssertionError, match="does not own"):
        halo.build_halo_info(partition, local_edges, group, edge_shard_sizes=partition.edge_splits, debug=True)
