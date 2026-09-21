# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Directed processor equivalence tests; run with ``--distributed``."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from distributed_runner import run_distributed_test

from anemoi.models.distributed.halo import build_halo_info
from anemoi.models.distributed.halo import verify_halo_info
from anemoi.models.distributed.khop_edges import build_graph_partition
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.processor import GraphTransformerProcessor
from anemoi.models.layers.utils import load_layer_kernels


def _test_processor_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    checkpoint: bool,
    sharded_edges: bool,
) -> None:
    torch.manual_seed(42)
    num_nodes = 3 * world_size + 1
    edge_index = torch.tensor(
        [[src, dst] for dst in range(num_nodes) for src in (dst, (dst + 2) % num_nodes)],
        device=device,
    ).T.contiguous()
    partition = build_graph_partition(edge_index, world_size, (num_nodes, num_nodes))
    reference = GraphTransformerProcessor(
        num_layers=2,
        num_channels=8,
        num_chunks=2,
        num_heads=2,
        mlp_hidden_ratio=2,
        edge_dim=3,
        graph_attention_backend="pyg",
        gradient_checkpointing=checkpoint,
        layer_kernels=load_layer_kernels(instance=False),
    ).to(device)
    processor = deepcopy(reference)
    full_x = torch.randn(num_nodes, 8, device=device, requires_grad=True)
    full_edges = torch.randn(edge_index.size(1), 3, device=device, requires_grad=True)
    expected = reference(full_x, 1, GraphShardInfo(), full_edges, edge_index)
    expected.square().sum().backward()

    local_x = full_x.detach().split(partition.dst_splits)[rank].clone().requires_grad_()
    local_edges = full_edges.detach()
    local_edge_index = edge_index
    if sharded_edges:
        local_edges = local_edges.split(partition.edge_splits)[rank]
        local_edge_index = edge_index.split(partition.edge_splits, dim=1)[rank].contiguous()
    local_edges = local_edges.clone().requires_grad_()
    shard_info = GraphShardInfo(
        nodes=partition.dst_splits,
        edges=partition.edge_splits if sharded_edges else None,
    )
    with (
        patch("anemoi.models.layers.processor.ANEMOI_DEBUG_SHARDING", True),
        patch("anemoi.models.layers.processor.build_halo_info", wraps=build_halo_info) as build,
        patch("anemoi.models.distributed.halo.verify_halo_info", wraps=verify_halo_info) as verify,
    ):
        actual = processor(local_x, 1, shard_info, local_edges, local_edge_index, group)
        torch.testing.assert_close(actual, expected.split(partition.dst_splits)[rank], atol=1e-6, rtol=1e-5)
        actual.square().sum().backward()
        with torch.no_grad():
            repeated = processor(local_x, 1, shard_info, local_edges, local_edge_index, group)
        torch.testing.assert_close(repeated, actual)
        assert build.call_count == 1
        assert verify.call_count == 1

    torch.testing.assert_close(local_x.grad, full_x.grad.split(partition.dst_splits)[rank], atol=1e-5, rtol=1e-5)
    expected_edge_grad = full_edges.grad.split(partition.edge_splits)[rank] if sharded_edges else full_edges.grad
    torch.testing.assert_close(local_edges.grad, expected_edge_grad, atol=1e-5, rtol=1e-5)
    for actual_param, expected_param in zip(processor.parameters(), reference.parameters()):
        assert actual_param.grad is not None
        assert expected_param.grad is not None
        dist.all_reduce(actual_param.grad, group=group)
        torch.testing.assert_close(actual_param.grad, expected_param.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.distributed
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("sharded_edges", [False, True])
def test_processor_halo_matches_full_graph(
    distributed_backend: str, distributed_world_size: int, checkpoint: bool, sharded_edges: bool
) -> None:
    run_distributed_test(
        _test_processor_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        checkpoint=checkpoint,
        sharded_edges=sharded_edges,
    )
