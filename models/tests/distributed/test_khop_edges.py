# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch

from anemoi.models.distributed.khop_edges import drop_unconnected_src_nodes
from anemoi.models.distributed.khop_edges import sort_edges_1hop_chunks
from anemoi.models.distributed.khop_edges import sort_edges_1hop_sharding


def _edge_triplet_set(edge_index: torch.Tensor, edge_attr: torch.Tensor) -> set[tuple[int, int, int]]:
    triplets = set()
    for idx in range(edge_index.shape[1]):
        src = int(edge_index[0, idx].item())
        dst = int(edge_index[1, idx].item())
        attr = int(edge_attr[idx, 0].item())
        triplets.add((src, dst, attr))
    return triplets


def _build_bipartite_graph() -> tuple[tuple[int, int], torch.Tensor, torch.Tensor]:
    num_nodes = (5, 6)
    edges = [
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (0, 5),
    ]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.arange(len(edges), dtype=torch.float32).unsqueeze(1)
    return num_nodes, edge_attr, edge_index


def test_sort_edges_1hop_chunks_bipartite_preserves_edges_and_attrs() -> None:
    num_nodes, edge_attr, edge_index = _build_bipartite_graph()
    num_chunks = 2
    node_chunks = torch.arange(num_nodes[1]).tensor_split(num_chunks)

    edge_attr_chunks, edge_index_chunks = sort_edges_1hop_chunks(
        num_nodes=num_nodes,
        edge_attr=edge_attr,
        edge_index=edge_index,
        num_chunks=num_chunks,
        relabel_dst_nodes=False,
    )

    assert len(edge_attr_chunks) == num_chunks
    assert len(edge_index_chunks) == num_chunks

    # Each chunk should only contain incoming edges for its destination node chunk.
    for chunk_id, edge_index_chunk in enumerate(edge_index_chunks):
        dst_nodes = edge_index_chunk[1] if edge_index_chunk.numel() > 0 else torch.tensor([], dtype=torch.long)
        assert torch.isin(dst_nodes, node_chunks[chunk_id]).all()

    original_set = _edge_triplet_set(edge_index, edge_attr)
    reconstructed_set = _edge_triplet_set(torch.cat(edge_index_chunks, dim=1), torch.cat(edge_attr_chunks, dim=0))
    assert reconstructed_set == original_set


def test_sort_edges_1hop_chunks_relabels_destinations_per_chunk() -> None:
    num_nodes, edge_attr, edge_index = _build_bipartite_graph()
    num_chunks = 2
    node_chunks = torch.arange(num_nodes[1]).tensor_split(num_chunks)

    edge_attr_chunks, edge_index_chunks = sort_edges_1hop_chunks(
        num_nodes=num_nodes,
        edge_attr=edge_attr,
        edge_index=edge_index,
        num_chunks=num_chunks,
        relabel_dst_nodes=True,
    )

    relabeled_global_triplets = set()
    for chunk_id, (edge_attr_chunk, edge_index_chunk) in enumerate(
        zip(edge_attr_chunks, edge_index_chunks, strict=True)
    ):
        if edge_index_chunk.numel() == 0:
            continue
        dst_relabelled = edge_index_chunk[1]
        assert int(dst_relabelled.min()) >= 0
        assert int(dst_relabelled.max()) < len(node_chunks[chunk_id])

        dst_global = dst_relabelled + node_chunks[chunk_id][0]
        edge_index_global = torch.stack([edge_index_chunk[0], dst_global], dim=0)
        relabeled_global_triplets |= _edge_triplet_set(edge_index_global, edge_attr_chunk)

    original_set = _edge_triplet_set(edge_index, edge_attr)
    assert relabeled_global_triplets == original_set


def test_sort_edges_1hop_sharding_without_group_returns_identity() -> None:
    num_nodes, edge_attr, edge_index = _build_bipartite_graph()
    sorted_attr, sorted_index, attr_shapes, index_shapes = sort_edges_1hop_sharding(
        num_nodes=num_nodes,
        edge_attr=edge_attr,
        edge_index=edge_index,
        mgroup=None,
    )

    torch.testing.assert_close(sorted_attr, edge_attr)
    torch.testing.assert_close(sorted_index, edge_index)
    assert attr_shapes == []
    assert index_shapes == []


def test_drop_unconnected_src_nodes_relabels_and_filters() -> None:
    x_src = torch.tensor(
        [
            [0.1, 0.0],
            [1.0, 1.1],
            [2.0, 2.1],
            [3.0, 3.1],
            [4.0, 4.1],
        ],
        dtype=torch.float32,
    )
    # Source node 2 is unconnected and should be dropped.
    edge_index = torch.tensor(
        [
            [1, 3, 4, 1],
            [0, 1, 2, 2],
        ],
        dtype=torch.long,
    )

    reduced_x_src, relabeled_edge_index, connected_src = drop_unconnected_src_nodes(
        x_src=x_src,
        edge_index=edge_index,
        num_nodes=(5, 3),
    )

    expected_connected = torch.tensor([1, 3, 4], dtype=torch.long)
    torch.testing.assert_close(connected_src, expected_connected)
    torch.testing.assert_close(reduced_x_src, x_src[expected_connected])

    # Expect source relabel mapping 1->0, 3->1, 4->2.
    expected_relabeled = torch.tensor(
        [
            [0, 1, 2, 0],
            [0, 1, 2, 2],
        ],
        dtype=torch.long,
    )
    torch.testing.assert_close(relabeled_edge_index, expected_relabeled)
