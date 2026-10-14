# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Mapper halo and full-source synchronization equivalence tests."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from distributed_runner import run_distributed_test

from anemoi.models.distributed.halo import build_halo_info_bipartite
from anemoi.models.distributed.khop_edges import build_graph_partition
from anemoi.models.distributed.khop_edges import build_graph_partition_from_shard_info
from anemoi.models.distributed.khop_edges import shard_graph_to_local
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.layers.mapper import GraphTransformerBackwardMapper
from anemoi.models.layers.mapper import GraphTransformerForwardMapper
from anemoi.models.layers.utils import load_layer_kernels


def _test_mapper_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    decoder: bool,
    checkpoint: bool,
) -> None:
    torch.manual_seed(42)
    num_src, num_dst = 2 * world_size + 1, 3 * world_size + 1
    edge_index = torch.tensor(
        [[src, dst] for dst in range(num_dst) for src in (dst % num_src, (dst + 2) % num_src)],
        device=device,
    ).T.contiguous()
    partition = build_graph_partition(edge_index, world_size, (num_src, num_dst))
    # Explicitly uneven source ownership, independent of destination ownership.
    src_splits = [1] * (world_size - 1) + [num_src - world_size + 1]
    mapper_type = GraphTransformerBackwardMapper if decoder else GraphTransformerForwardMapper
    reference = mapper_type(
        in_channels_src=8,
        in_channels_dst=3,
        num_channels=8,
        out_channels_dst=4 if decoder else None,
        num_chunks=2,
        num_heads=2,
        mlp_hidden_ratio=2,
        edge_dim=2,
        graph_attention_backend="pyg",
        gradient_checkpointing=checkpoint,
        layer_kernels=load_layer_kernels(instance=False),
    ).to(device)
    full_src = torch.randn(num_src, 8, device=device, requires_grad=True)
    full_dst = torch.randn(num_dst, 3, device=device, requires_grad=True)
    full_edges = torch.randn(edge_index.size(1), 2, device=device, requires_grad=True)
    output = reference((full_src, full_dst), 1, BipartiteGraphShardInfo(), full_edges, edge_index)
    expected = output if decoder else output[1]
    expected.square().sum().backward()

    for sharded_nodes, sharded_edges in ((False, False), (True, False), (True, True)):
        for use_halo in (False, True):
            mapper = deepcopy(reference)
            mapper.zero_grad(set_to_none=True)
            mapper.use_halo_exchange = use_halo
            inputs = [full_src, full_dst, full_edges]
            splits = [src_splits, partition.dst_splits, partition.edge_splits]
            sharded = [sharded_nodes, sharded_nodes, sharded_edges]
            local_inputs = [
                (tensor.detach().split(sizes)[rank] if is_sharded else tensor.detach()).clone().requires_grad_()
                for tensor, sizes, is_sharded in zip(inputs, splits, sharded)
            ]
            src, dst, edges = local_inputs
            local_edge_index = (
                edge_index.split(partition.edge_splits, dim=1)[rank].contiguous() if sharded_edges else edge_index
            )
            shard_info = BipartiteGraphShardInfo(
                src_nodes=src_splits if sharded_nodes else None,
                dst_nodes=partition.dst_splits if sharded_nodes else None,
                edges=partition.edge_splits if sharded_edges else None,
            )
            with patch(
                "anemoi.models.layers.mapper.build_halo_info_bipartite", wraps=build_halo_info_bipartite
            ) as build:
                output = mapper((src, dst), 1, shard_info, edges, local_edge_index, group, keep_x_dst_sharded=True)
                actual = output if decoder else output[1]
                torch.testing.assert_close(actual, expected.split(partition.dst_splits)[rank], atol=1e-6, rtol=1e-5)
                actual.square().sum().backward()
                with torch.no_grad():
                    output = mapper((src, dst), 1, shard_info, edges, local_edge_index, group, keep_x_dst_sharded=False)
                gathered = output if decoder else output[1]
                torch.testing.assert_close(gathered, expected, atol=1e-6, rtol=1e-5)
                assert build.call_count == int(use_halo)

            for actual_input, full_input, sizes, is_sharded in zip(local_inputs, inputs, splits, sharded):
                expected_grad = full_input.grad.split(sizes)[rank] if is_sharded else full_input.grad
                torch.testing.assert_close(actual_input.grad, expected_grad, atol=1e-5, rtol=1e-5)
            for actual_param, expected_param in zip(mapper.parameters(), reference.parameters()):
                assert actual_param.grad is not None
                assert expected_param.grad is not None
                dist.all_reduce(actual_param.grad, group=group)
                torch.testing.assert_close(actual_param.grad, expected_param.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.distributed
@pytest.mark.parametrize("decoder", [False, True])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_mapper_halo_matches_full_graph(
    distributed_backend: str, distributed_world_size: int, decoder: bool, checkpoint: bool
) -> None:
    run_distributed_test(
        _test_mapper_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        decoder=decoder,
        checkpoint=checkpoint,
    )


def _test_conditioning_rank(*, rank: int, world_size: int, device: torch.device, group: dist.ProcessGroup) -> None:
    num_src, num_dst = 2 * world_size + 1, 3 * world_size + 1
    edge_index = torch.tensor([[(dst + 2) % num_src, dst] for dst in range(num_dst)], device=device).T.contiguous()
    src_splits = [1] * (world_size - 1) + [num_src - world_size + 1]
    partition = build_graph_partition(edge_index, world_size, (num_src, num_dst))
    full_src = torch.arange(num_src, device=device, dtype=torch.float32).unsqueeze(1)
    full_dst = torch.arange(num_dst, device=device, dtype=torch.float32).unsqueeze(1)
    expected_inputs = [tensor.clone().requires_grad_() for tensor in (full_src, full_dst, full_src + 10, full_dst + 20)]
    src, dst, cond_src, cond_dst = expected_inputs
    expected = (src + cond_src)[edge_index[0]] + (dst + cond_dst)[edge_index[1]]
    expected.square().sum().backward()
    for sharded_nodes, use_halo in ((False, False), (True, False), (True, True)):
        sizes = [src_splits, partition.dst_splits, src_splits, partition.dst_splits]
        inputs = [
            (tensor.detach().split(split)[rank] if sharded_nodes else tensor.detach()).clone().requires_grad_()
            for tensor, split in zip(expected_inputs, sizes)
        ]
        src, dst, cond_src, cond_dst = inputs
        shard_info = BipartiteGraphShardInfo(
            src_nodes=src_splits if sharded_nodes else None,
            dst_nodes=partition.dst_splits if sharded_nodes else None,
        )
        partition = build_graph_partition_from_shard_info(edge_index, (src, dst), shard_info, group)
        info = build_halo_info_bipartite(partition, edge_index, group, debug=True) if use_halo else None
        (local_src, local_dst), _, local_edges, _, local_cond = shard_graph_to_local(
            partition,
            (src, dst),
            torch.ones(edge_index.size(1), 1, device=device),
            edge_index,
            shard_info,
            group,
            cond=(cond_src, cond_dst),
            halo_info=info,
        )
        assert local_cond is not None
        actual = (local_src + local_cond[0])[local_edges[0]] + (local_dst + local_cond[1])[local_edges[1]]
        torch.testing.assert_close(actual, expected.split(partition.edge_splits)[rank])
        actual.square().sum().backward()
        for actual_input, expected_input, split in zip(inputs, expected_inputs, sizes):
            expected_grad = expected_input.grad.split(split)[rank] if sharded_nodes else expected_input.grad
            torch.testing.assert_close(actual_input.grad, expected_grad)


@pytest.mark.distributed
def test_mapper_conditioning_localization(distributed_backend: str, distributed_world_size: int) -> None:
    run_distributed_test(_test_conditioning_rank, backend=distributed_backend, world_size=distributed_world_size)
