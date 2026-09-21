# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Halo metadata tests, with optional real collectives via ``--distributed``."""

from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
from distributed_runner import run_distributed_test

from anemoi.models.distributed import halo
from anemoi.models.distributed.khop_edges import GraphPartition

GRAPH_CASES = {
    "symmetric": (
        None,
        [2, 3, 1],
        [(0, 2), (2, 0), (1, 5), (5, 1), (4, 5), (5, 4), (2, 2)],
    ),
    "directed": (
        None,
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
    "local-only": (None, [2, 3, 1], [(0, 1), (1, 0), (2, 4), (5, 5)]),
    "empty-edges": (None, [2, 3, 1], []),
    "single-rank": (None, [3], [(0, 2), (1, 1)]),
    "single-rank-bipartite": ([2], [3], [(0, 2), (1, 1)]),
}


def _make_graph(case: str, device: torch.device) -> tuple[GraphPartition, torch.Tensor]:
    src_splits, dst_splits, edges = GRAPH_CASES[case]
    edge_index = torch.tensor(sorted(edges, key=lambda edge: edge[1]), dtype=torch.long, device=device)
    edge_index = edge_index.reshape(-1, 2).T.contiguous()
    dst_owners = [rank for rank, size in enumerate(dst_splits) for _ in range(size)]
    edge_splits = [sum(dst_owners[dst] == rank for _, dst in edges) for rank in range(len(dst_splits))]
    partition = GraphPartition(
        num_nodes=(sum(dst_splits if src_splits is None else src_splits), sum(dst_splits)),
        num_edges=len(edges),
        num_parts=len(dst_splits),
        dst_splits=dst_splits,
        edge_splits=edge_splits,
        src_splits=src_splits,
    )
    return partition, edge_index


def _assert_halo_info(
    info: halo.HaloInfo,
    partition: GraphPartition,
    edge_index: torch.Tensor,
    rank: int,
    debug: bool,
) -> None:
    """Check exact metadata against a node-ownership reference, not reverse edges."""
    src_splits = partition.dst_splits if partition.src_splits is None else partition.src_splits
    src_owners = [peer for peer, size in enumerate(src_splits) for _ in range(size)]
    dst_owners = [peer for peer, size in enumerate(partition.dst_splits) for _ in range(size)]
    edges = edge_index.T.tolist()
    local_src_ids = [node for node, owner in enumerate(src_owners) if owner == rank]
    local_dst_ids = [node for node, owner in enumerate(dst_owners) if owner == rank]
    expected_recv = [
        sorted({src for src, dst in edges if dst_owners[dst] == rank and src_owners[src] == peer and peer != rank})
        for peer in range(partition.num_parts)
    ]
    expected_send = [
        sorted({src for src, dst in edges if src_owners[src] == rank and dst_owners[dst] == peer and peer != rank})
        for peer in range(partition.num_parts)
    ]
    halo_ids = [node for peer_nodes in expected_recv for node in peer_nodes]
    src_to_local = {node: index for index, node in enumerate(local_src_ids + halo_ids)}
    dst_to_local = {node: index for index, node in enumerate(local_dst_ids)}
    expected_edges = [(src_to_local[src], dst_to_local[dst]) for src, dst in edges if dst_owners[dst] == rank]
    expected_edge_index = edge_index.new_tensor(expected_edges).reshape(-1, 2).T

    assert info.num_local_nodes == info.num_local_src_nodes == len(local_src_ids)
    assert info.num_local_dst_nodes == info.local_dst_nodes == len(local_dst_ids)
    assert info.num_halo_nodes == len(halo_ids)
    assert info.total_nodes == info.total_src_nodes == len(local_src_ids) + len(halo_ids)
    assert info.recv_counts == tuple(map(len, expected_recv))
    assert info.send_counts == tuple(map(len, expected_send))
    assert len(info.send_indices) == partition.num_parts
    for actual, nodes in zip(info.send_indices, expected_send):
        torch.testing.assert_close(actual, edge_index.new_tensor([src_to_local[node] for node in nodes]))
    if debug:
        assert info.recv_global_ids is not None
        assert len(info.recv_global_ids) == partition.num_parts
        for actual, nodes in zip(info.recv_global_ids, expected_recv):
            torch.testing.assert_close(actual, edge_index.new_tensor(nodes))
    else:
        assert info.recv_global_ids is None
    torch.testing.assert_close(info.edge_index_local, expected_edge_index)


@pytest.mark.parametrize("case", GRAPH_CASES)
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("debug", [False, True])
def test_build_halo_info(case: str, sharded: bool, debug: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    partition, edge_index = _make_graph(case, torch.device("cpu"))
    original_edges = edge_index.clone()
    group = Mock(spec=dist.ProcessGroup)
    group.size.return_value = partition.num_parts
    builders = [halo.build_halo_info]
    if partition.src_splits is not None:
        builders.append(halo.build_halo_info_bipartite)

    for rank in range(partition.num_parts):
        local_edges = edge_index.split(partition.edge_splits, dim=1)[rank].contiguous()
        original_local_edges = local_edges.clone()
        monkeypatch.setattr(dist, "get_rank", Mock(return_value=rank))
        for build in builders:
            gather = Mock(return_value=edge_index)
            shard = Mock(return_value=local_edges)
            verify = Mock()
            monkeypatch.setattr(halo, "gather_tensor", gather)
            monkeypatch.setattr(halo, "shard_tensor", shard)
            monkeypatch.setattr(halo, "verify_halo_info", verify)
            info = build(
                partition,
                local_edges if sharded else edge_index,
                group,
                edge_shard_sizes=partition.edge_splits if sharded else None,
                debug=debug,
            )
            _assert_halo_info(info, partition, edge_index, rank, debug)
            torch.testing.assert_close(edge_index, original_edges)
            torch.testing.assert_close(local_edges, original_local_edges)
            if sharded:
                gather.assert_called_once_with(local_edges, 1, partition.edge_splits, group)
                shard.assert_not_called()
            else:
                shard.assert_called_once_with(edge_index, 1, partition.edge_splits, group)
                gather.assert_not_called()
            if debug:
                verify.assert_called_once_with(info, partition, group)
            else:
                verify.assert_not_called()


def test_bipartite_requires_source_partition() -> None:
    partition, edge_index = _make_graph("directed", torch.device("cpu"))
    group = Mock(spec=dist.ProcessGroup)
    group.size.return_value = partition.num_parts
    with pytest.raises(AssertionError, match="Bipartite partition must have src_splits"):
        halo.build_halo_info_bipartite(partition, edge_index, group)


def test_partition_must_match_group_size(monkeypatch: pytest.MonkeyPatch) -> None:
    partition, edge_index = _make_graph("directed", torch.device("cpu"))
    group = Mock(spec=dist.ProcessGroup)
    group.size.return_value = partition.num_parts + 1
    monkeypatch.setattr(dist, "get_rank", Mock(return_value=0))
    with pytest.raises(AssertionError, match="Partition num_parts"):
        halo.build_halo_info(partition, edge_index, group)


def _test_halo_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
) -> None:
    for case in GRAPH_CASES:
        partition, edge_index = _make_graph(case, device)
        if partition.num_parts == 1:
            continue
        assert partition.num_parts == world_size
        build = halo.build_halo_info if partition.src_splits is None else halo.build_halo_info_bipartite
        local_edges = edge_index.split(partition.edge_splits, dim=1)[rank].contiguous()
        for sharded in (False, True):
            for debug in (False, True):
                info = build(
                    partition,
                    local_edges if sharded else edge_index,
                    group,
                    edge_shard_sizes=partition.edge_splits if sharded else None,
                    debug=debug,
                )
                _assert_halo_info(info, partition, edge_index, rank, debug)


@pytest.mark.distributed
def test_halo_collectives(distributed_backend: str) -> None:
    # Three ranks exercise send-only, receive-only, and intermediate ownership.
    run_distributed_test(_test_halo_rank, backend=distributed_backend, world_size=3)
