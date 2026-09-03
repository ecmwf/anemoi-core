# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Distributed tests for graph halo metadata and feature exchange."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.distributed as dist
from distributed_runner import run_distributed_test

from anemoi.models.distributed.graph import halo_exchange
from anemoi.models.distributed.halo import build_halo_info_bipartite
from anemoi.models.distributed.khop_edges import GraphPartition
from anemoi.models.distributed.khop_edges import shard_graph_to_local
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.layers.mapper import GraphTransformerForwardMapper
from anemoi.models.layers.utils import load_layer_kernels


def _directed_ring_edges(world_size: int, device: torch.device) -> torch.Tensor:
    edges = []
    for rank in range(world_size):
        previous_rank = (rank - 1) % world_size
        next_rank = (rank + 1) % world_size
        edges.extend(
            [
                (3 * rank, 2 * rank),
                (3 * previous_rank + 1, 2 * rank),
                (3 * rank + 1, 2 * rank + 1),
                (3 * next_rank + 2, 2 * rank + 1),
            ]
        )
    return torch.tensor(edges, dtype=torch.long, device=device).T.contiguous()


def _test_bipartite_halo_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
) -> None:
    edges = _directed_ring_edges(world_size, device)
    edge_splits = [4] * world_size
    local_edges = edges[:, rank * 4 : (rank + 1) * 4]
    partition = GraphPartition(
        num_nodes=(3 * world_size, 2 * world_size),
        num_edges=edges.size(1),
        num_parts=world_size,
        src_splits=[3] * world_size,
        dst_splits=[2] * world_size,
        edge_splits=edge_splits,
    )

    halo_info = build_halo_info_bipartite(
        partition,
        local_edges,
        group,
        edge_shard_sizes=edge_splits,
        debug=True,
    )

    assert halo_info.num_local_src_nodes == 3
    assert halo_info.local_dst_nodes == 2
    assert halo_info.edge_index_local[1].tolist() == [0, 0, 1, 1]
    assert halo_info.edge_index_local[0].max().item() < halo_info.total_src_nodes

    full_features = torch.arange(3 * world_size, dtype=torch.float32, device=device).unsqueeze(-1)
    local_features = full_features[rank * 3 : (rank + 1) * 3].clone().requires_grad_()
    features_with_halo = halo_exchange(local_features, halo_info, group)

    recv_ids = torch.cat(halo_info.recv_global_ids)
    expected_features = torch.cat((local_features.detach(), full_features[recv_ids]))
    torch.testing.assert_close(features_with_halo, expected_features)

    full_dst = torch.arange(2 * world_size, dtype=torch.float32, device=device).unsqueeze(-1)
    (_, localized_dst), _, _, _, localized_cond = shard_graph_to_local(
        partition,
        (local_features.detach(), full_dst),
        torch.zeros(local_edges.size(1), 1, device=device),
        local_edges,
        BipartiteGraphShardInfo(src_nodes=[3] * world_size, edges=edge_splits),
        group,
        cond=(local_features.detach(), full_dst),
        halo_info=halo_info,
    )
    expected_dst = full_dst[rank * 2 : (rank + 1) * 2]
    torch.testing.assert_close(localized_dst, expected_dst)
    torch.testing.assert_close(localized_cond[0], expected_features)
    torch.testing.assert_close(localized_cond[1], expected_dst)

    features_with_halo.sum().backward()
    expected_gradient = torch.ones_like(local_features)
    for send_indices in halo_info.send_indices:
        expected_gradient.index_add_(0, send_indices, torch.ones_like(local_features[send_indices]))
    torch.testing.assert_close(local_features.grad, expected_gradient)


@pytest.mark.distributed
def test_bipartite_halo_exchange(
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    run_distributed_test(
        _test_bipartite_halo_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
    )


def _test_mapper_halo_parity_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    use_halo_exchange: bool,
) -> None:
    del rank
    torch.manual_seed(42)
    edges = _directed_ring_edges(world_size, device)
    edge_attr = torch.randn(edges.size(1), 3, device=device)
    x_src = torch.randn(3 * world_size, 5, device=device)
    x_dst = torch.randn(2 * world_size, 4, device=device)

    mapper = GraphTransformerForwardMapper(
        in_channels_src=5,
        in_channels_dst=4,
        hidden_dim=8,
        num_chunks=2,
        num_heads=2,
        mlp_hidden_ratio=2,
        edge_dim=3,
        gradient_checkpointing=True,
        layer_kernels=load_layer_kernels(instance=False),
        shard_strategy="edges",
        use_halo_exchange=use_halo_exchange,
        graph_attention_backend="pyg",
    ).to(device)
    reference_mapper = copy.deepcopy(mapper)

    reference_src = x_src.clone().requires_grad_()
    _, reference_output = reference_mapper(
        (reference_src, x_dst),
        batch_size=1,
        shard_info=BipartiteGraphShardInfo(
            src_nodes=[x_src.size(0)],
            dst_nodes=[x_dst.size(0)],
            edges=[edges.size(1)],
        ),
        edge_attr=edge_attr,
        edge_index=edges,
    )
    reference_output.sum().backward()

    distributed_src = x_src.clone().requires_grad_()
    _, distributed_output = mapper(
        (distributed_src, x_dst),
        batch_size=1,
        shard_info=BipartiteGraphShardInfo(),
        edge_attr=edge_attr,
        edge_index=edges,
        model_comm_group=group,
        keep_x_dst_sharded=False,
    )
    cached_halo_info = mapper._cached_halo_info
    cached_halo_partition = mapper._cached_halo_partition
    with torch.no_grad():
        _, repeated_output = mapper(
            (distributed_src.detach(), x_dst),
            batch_size=1,
            shard_info=BipartiteGraphShardInfo(),
            edge_attr=edge_attr,
            edge_index=edges,
            model_comm_group=group,
            keep_x_dst_sharded=False,
        )

    if use_halo_exchange:
        assert mapper._cached_halo_info is cached_halo_info
        assert mapper._cached_halo_partition is cached_halo_partition
    else:
        assert mapper._cached_halo_info is None
        assert mapper._cached_halo_partition is None
    torch.testing.assert_close(repeated_output, reference_output, atol=1e-5, rtol=1e-5)
    distributed_output.sum().backward()

    torch.testing.assert_close(distributed_output, reference_output, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(distributed_src.grad, reference_src.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.distributed
@pytest.mark.parametrize("use_halo_exchange", [False, True])
def test_graphtransformer_mapper_halo_parity(
    distributed_backend: str,
    distributed_world_size: int,
    use_halo_exchange: bool,
) -> None:
    run_distributed_test(
        _test_mapper_halo_parity_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        use_halo_exchange=use_halo_exchange,
    )