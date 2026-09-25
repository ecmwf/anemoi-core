# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Distributed tests for the graph transformer processor with halo exchange.

Each rank runs the processor on its own slice of the nodes and compares the
result with the same processor run on the whole graph in a single process.

These tests are skipped by default. Pass ``--distributed`` to run them. Use
``--distributed-backend`` and ``--distributed-world-size`` to select the backend
and rank count.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
from distributed_runner import run_distributed_test

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.khop_edges import shard_edges_1hop
from anemoi.models.distributed.khop_edges import sort_edge_index_by_dst
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers import processor as processor_module
from anemoi.models.layers.processor import GraphTransformerProcessor
from anemoi.models.layers.utils import load_layer_kernels

NUM_NODES = 57
NUM_CHANNELS = 32
EDGE_DIM = 5


def _directed_edges(generator: torch.Generator, world_size: int) -> torch.Tensor:
    """Random edges plus a block of one-way edges from the first third of the nodes to the last third."""
    src = torch.randint(0, NUM_NODES, (400,), generator=generator)
    dst = torch.randint(0, NUM_NODES, (400,), generator=generator)
    third = NUM_NODES // 3
    src[:100] = torch.randint(0, third, (100,), generator=generator)
    dst[:100] = torch.randint(NUM_NODES - third, NUM_NODES, (100,), generator=generator)
    return torch.stack([src, dst])


def _sparse_request_edges(generator: torch.Generator, world_size: int) -> torch.Tensor:
    """Edges inside each rank's own nodes, plus edges from the last rank's nodes into rank 0's nodes.

    Only rank 0 needs nodes from another rank, so every other rank requests nothing.
    """
    sizes = get_balanced_partition_sizes(NUM_NODES, world_size)
    starts = [sum(sizes[:rank]) for rank in range(world_size)]
    edges = []
    for start, size in zip(starts, sizes):
        src = torch.randint(start, start + size, (30,), generator=generator)
        dst = torch.randint(start, start + size, (30,), generator=generator)
        edges.append(torch.stack([src, dst]))
    src = torch.randint(starts[-1], starts[-1] + sizes[-1], (20,), generator=generator)
    dst = torch.randint(0, sizes[0], (20,), generator=generator)
    edges.append(torch.stack([src, dst]))
    return torch.cat(edges, dim=1)


GRAPHS = {"directed": _directed_edges, "sparse-requests": _sparse_request_edges}


def _build_case(graph: str, world_size: int, device: torch.device):
    generator = torch.Generator().manual_seed(0)
    edge_index = GRAPHS[graph](generator, world_size)
    edge_index, perm = sort_edge_index_by_dst(edge_index, max_value=NUM_NODES)
    edge_attr = torch.randn(edge_index.size(1), EDGE_DIM, generator=generator)[perm]
    x = torch.randn(NUM_NODES, NUM_CHANNELS, generator=generator)
    loss_weights = torch.randn(NUM_NODES, NUM_CHANNELS, generator=generator)

    torch.manual_seed(1)
    processor = GraphTransformerProcessor(
        num_layers=3,
        num_channels=NUM_CHANNELS,
        num_chunks=1,
        num_heads=4,
        mlp_hidden_ratio=2,
        qk_norm=True,
        cpu_offload=False,
        layer_kernels=load_layer_kernels(instance=False),
        graph_attention_backend="pyg",
        edge_dim=EDGE_DIM,
        shard_strategy="edges",
    ).to(device)
    return processor, x.to(device), edge_attr.to(device), edge_index.to(device), loss_weights.to(device)


def _pairwise_send_counts(send_counts: tuple[int, ...], device: torch.device, group: dist.ProcessGroup) -> torch.Tensor:
    """Matrix whose entry (i, j) is the number of nodes rank i sends to rank j."""
    local = torch.tensor(send_counts, dtype=torch.long, device=device)
    rows = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(rows, local, group=group)
    return torch.stack(rows)


def _test_processor_matches_single_process_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    graph: str,
    pre_sharded: bool,
    debug: bool,
) -> None:
    processor_module.ANEMOI_DEBUG_SHARDING = debug
    processor, x, edge_attr, edge_index, loss_weights = _build_case(graph, world_size, device)

    # Reference: the whole graph in one process.
    x_full = x.clone().requires_grad_(True)
    y_full = processor(x_full, 1, GraphShardInfo(), edge_attr, edge_index, model_comm_group=None)
    (y_full * loss_weights).sum().backward()
    param_grads_full = {name: param.grad.clone() for name, param in processor.named_parameters()}
    processor.zero_grad(set_to_none=True)

    sizes = get_balanced_partition_sizes(NUM_NODES, world_size)
    start, stop = sum(sizes[:rank]), sum(sizes[: rank + 1])
    x_local = x[start:stop].clone().requires_grad_(True)
    edge_shard_sizes = None
    if pre_sharded:
        edge_attr, edge_index, edge_shard_sizes = shard_edges_1hop(edge_attr, edge_index, NUM_NODES, NUM_NODES, group)
    shard_info = GraphShardInfo(nodes=sizes, edges=edge_shard_sizes)

    y_local = processor(x_local, 1, shard_info, edge_attr, edge_index, model_comm_group=group)
    (y_local * loss_weights[start:stop]).sum().backward()

    tolerances = {"rtol": 1e-4, "atol": 1e-5}
    torch.testing.assert_close(y_local, y_full[start:stop], **tolerances)
    torch.testing.assert_close(x_local.grad, x_full.grad[start:stop], **tolerances)
    for name, param in processor.named_parameters():
        grad = param.grad.clone()
        dist.all_reduce(grad, group=group)
        torch.testing.assert_close(grad, param_grads_full[name], **tolerances, msg=f"gradient of {name}")

    halo_info = processor._cached_halo_info
    sends = _pairwise_send_counts(halo_info.send_counts, device, group)
    receives = _pairwise_send_counts(halo_info.recv_counts, device, group)
    torch.testing.assert_close(receives, sends.T)
    if graph == "directed":
        assert not torch.equal(sends, sends.T), "expected the directed graph to give uneven sends between ranks"
    else:
        assert (receives.sum(dim=1)[1:] == 0).all(), "expected only rank 0 to request nodes"

    with torch.no_grad():
        y_again = processor(x_local, 1, shard_info, edge_attr, edge_index, model_comm_group=group)
    assert processor._cached_halo_info is halo_info
    torch.testing.assert_close(y_again, y_local)


@pytest.mark.distributed
@pytest.mark.parametrize("graph", list(GRAPHS))
@pytest.mark.parametrize("pre_sharded", [False, True])
@pytest.mark.parametrize("debug", [False, True])
def test_processor_matches_single_process(
    distributed_backend: str, distributed_world_size: int, graph: str, pre_sharded: bool, debug: bool
) -> None:
    run_distributed_test(
        _test_processor_matches_single_process_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        graph=graph,
        pre_sharded=pre_sharded,
        debug=debug,
    )
