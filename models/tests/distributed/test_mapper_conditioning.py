# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Conditioning follows node features through mapper sharding and backward communication."""

from copy import deepcopy
from itertools import product
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.models.data import Batch
from anemoi.models.data import TensorLayout
from anemoi.models.data.sources import GriddedSource
from anemoi.models.data.sources import TabularSource
from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.khop_edges import build_graph_partition
from anemoi.models.distributed.khop_edges import shard_graph_to_local
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.layers.graph import NodeTrainableParameters
from anemoi.models.layers.mapper import GraphTransformerForwardMapper
from anemoi.models.layers.mapper import TransformerForwardMapper
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportModelEncProcDec

from ._distributed_runner import _run_distributed_test


def _check_mapper_conditioning(*, rank, world_size, device, group, kind) -> None:
    torch.manual_seed(42)
    channels = 4 * world_size
    config = dict(
        in_channels_src=4,
        in_channels_dst=3,
        num_channels=channels,
        num_heads=world_size,
        num_chunks=2,
        mlp_hidden_ratio=2,
        gradient_checkpointing=False,
        layer_kernels=load_layer_kernels(
            kernel_config={
                "LayerNorm": {
                    "_target_": "anemoi.models.layers.normalization.ConditionalLayerNorm",
                    "condition_shape": 2,
                    "zero_init": False,
                },
            },
            instance=False,
        ),
    )
    if kind == "transformer":
        mapper = TransformerForwardMapper(**config, attention_implementation="scaled_dot_product_attention")
    else:
        mapper = GraphTransformerForwardMapper(**config, edge_dim=1, shard_strategy=kind, graph_attention_backend="pyg")
    mapper = mapper.to(device)
    reference = deepcopy(mapper)

    n_src, n_dst = 2 * world_size + 1, 3 * world_size + 1
    src_sizes = get_balanced_partition_sizes(n_src, world_size)
    dst_sizes = get_balanced_partition_sizes(n_dst, world_size)
    dst_slice = slice(sum(dst_sizes[:rank]), sum(dst_sizes[: rank + 1]))
    # Different source subsets per destination exercise halo gathering and local chunk selection.
    dst_ids = torch.arange(n_dst, device=device).repeat_interleave(2)
    src_ids = (dst_ids + torch.arange(2, device=device).repeat(n_dst)) % n_src
    edge_index = torch.stack((src_ids, dst_ids))
    edge_attr = torch.randn(len(dst_ids), 1, device=device)
    values = [torch.randn(n, width, device=device) for n, width in ((n_src, 4), (n_dst, 3), (n_src, 2), (n_dst, 2))]
    weights = torch.randn(n_dst, channels, device=device)
    kwargs = dict(batch_size=1, edge_index=edge_index, edge_attr=edge_attr)

    for src_sharded, dst_sharded in product((False, True), repeat=2):
        full = [value.clone().requires_grad_() for value in values]
        expected = reference(
            tuple(full[:2]),
            shard_info=BipartiteGraphShardInfo(src_nodes=None, dst_nodes=None, edges=None),
            cond=tuple(full[2:]),
            **kwargs,
        )[1]
        (expected * weights).sum().backward()

        actual_inputs = [value.clone().requires_grad_() for value in values]
        layouts = (src_sizes if src_sharded else None, dst_sizes if dst_sharded else None)
        local = [
            value if sizes is None else shard_tensor(value, 0, sizes, group)
            for value, sizes in zip(actual_inputs, (*layouts, *layouts), strict=True)
        ]
        actual = mapper(
            tuple(local[:2]),
            shard_info=BipartiteGraphShardInfo(src_nodes=layouts[0], dst_nodes=layouts[1], edges=None),
            cond=tuple(local[2:]),
            model_comm_group=group,
            keep_x_dst_sharded=True,
            **kwargs,
        )[1]
        torch.testing.assert_close(actual, expected[dst_slice], atol=2e-5, rtol=2e-5)
        (actual * weights[dst_slice]).sum().backward()
        for value, expected_value in zip(actual_inputs, full, strict=True):
            torch.testing.assert_close(value.grad, expected_value.grad, atol=2e-5, rtol=2e-5)


@pytest.mark.distributed
@pytest.mark.parametrize("kind", ["edges", "heads", "transformer"])
def test_mapper_conditioning_matches_unsharded_outputs_and_gradients(
    distributed_backend, distributed_world_size, kind
) -> None:
    _run_distributed_test(
        _check_mapper_conditioning,
        backend=distributed_backend,
        world_size=distributed_world_size,
        kind=kind,
    )


def _check_invalid_conditioning_rows(*, rank, world_size, device, group) -> None:
    n_src, n_dst = 2 * world_size + 1, 3 * world_size + 1
    edge_index = torch.stack(
        (
            torch.arange(n_src, device=device).repeat(n_dst),
            torch.arange(n_dst, device=device).repeat_interleave(n_src),
        )
    )
    partition = build_graph_partition(edge_index, num_parts=world_size, num_nodes=(n_src, n_dst))
    local_dst = partition.dst_splits[rank]
    x = (torch.zeros(n_src, 4, device=device), torch.zeros(local_dst, 4, device=device))
    shard_info = BipartiteGraphShardInfo(src_nodes=None, dst_nodes=partition.dst_splits, edges=None)

    # Replicated sources and already-sharded destinations previously bypassed row-count validation.
    for side, row_delta in product(("Source", "Destination"), (-1, 1)):
        cond_rows = [n_src, local_dst]
        index = 0 if side == "Source" else 1
        expected_rows = cond_rows[index]
        cond_rows[index] += row_delta
        message = (
            f"{side} conditioning has {cond_rows[index]} rows, "
            f"but {side.lower()} node features have {expected_rows} rows"
        )
        with pytest.raises(ValueError, match=message):
            shard_graph_to_local(
                partition,
                x,
                torch.zeros(edge_index.shape[1], 1, device=device),
                edge_index,
                shard_info,
                group,
                cond=tuple(torch.zeros(rows, 2, device=device) for rows in cond_rows),
            )


@pytest.mark.distributed
def test_edge_sharding_rejects_mismatched_conditioning_rows(distributed_backend, distributed_world_size) -> None:
    _run_distributed_test(
        _check_invalid_conditioning_rows,
        backend=distributed_backend,
        world_size=distributed_world_size,
    )


def _check_transport_conditioning(*, rank, world_size, device, group) -> None:
    # Two observation windows, each with uneven partitions. Their sum is the flat node partition.
    window_sizes = get_balanced_partition_sizes(2 * world_size + 1, world_size)
    node_sizes = [2 * n for n in window_sizes]
    local_nodes = node_sizes[rank]
    total_nodes = sum(node_sizes)
    hidden_nodes = 3 * world_size + 1
    hidden_sizes = get_balanced_partition_sizes(hidden_nodes, world_size)
    node_start = sum(node_sizes[:rank])
    hidden_start = sum(hidden_sizes[:rank])

    model = AnemoiTransportModelEncProcDec.__new__(AnemoiTransportModelEncProcDec)
    torch.nn.Module.__init__(model)
    model._graph_name_hidden = "hidden"
    model._graph_data = {"hidden": SimpleNamespace(num_nodes=hidden_nodes)}
    model.noise_embedder = torch.nn.Identity()
    model.noise_cond_mlp = torch.nn.Linear(1, 2, bias=False).to(device)
    with torch.no_grad():
        model.noise_cond_mlp.weight.copy_(torch.tensor([[2.0], [3.0]], device=device))

    common = dict(name="data", variables=["a"], statistics={})
    gridded = GriddedSource(
        **common,
        data=torch.zeros(1, 1, 1, local_nodes, 1, device=device),
        coordinates=torch.zeros(local_nodes, 2, device=device),
        layout=TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4),
        shard_sizes=node_sizes,
    )
    tabular = TabularSource(
        **common,
        data=[torch.zeros(1, local_nodes, 1, device=device)],
        coordinates=[torch.zeros(local_nodes, 2, device=device)],
        timedeltas=[torch.zeros(local_nodes, device=device)],
        boundaries=[(slice(0, window_sizes[rank]), slice(window_sizes[rank], local_nodes))],
        layout=TensorLayout(ensemble=0, grid=1, variables=2),
        shard_sizes=[[window_sizes, window_sizes]],
    )
    for source in (gridded, tabular):
        model.zero_grad()
        fwd, proc, bwd = model._build_conditioning_kwargs(
            Batch({"data": source}), {"data": torch.ones(1, 1, 1, 1, 1, device=device)}, group
        )
        data_cond, hidden_cond = fwd["data"]["cond"]
        assert bwd["data"]["cond"][1] is data_cond
        assert bwd["data"]["cond"][0] is proc["cond"] is hidden_cond
        torch.testing.assert_close(data_cond, torch.tensor([2.0, 3.0], device=device).expand(local_nodes, 2))
        torch.testing.assert_close(hidden_cond, torch.tensor([2.0, 3.0], device=device).expand(hidden_sizes[rank], 2))
        data_weights = torch.arange(node_start + 1, node_start + local_nodes + 1, device=device)[:, None]
        hidden_weights = torch.arange(hidden_start + 1, hidden_start + hidden_sizes[rank] + 1, device=device)[:, None]
        ((data_cond * data_weights).sum() + (hidden_cond * hidden_weights).sum()).backward()
        expected_gradient = total_nodes * (total_nodes + 1) / 2 + hidden_nodes * (hidden_nodes + 1) / 2
        torch.testing.assert_close(
            model.noise_cond_mlp.weight.grad, torch.full((2, 1), expected_gradient, device=device)
        )

    # Static trainable attributes are global; input assembly must select this rank's rows.
    graph = HeteroData()
    graph["data"].x = torch.zeros(total_nodes, 2, device=device)
    model.node_attributes = NodeTrainableParameters({"data": 2}, graph).to(device)
    coords, latent, _, sizes, _, _ = model._assemble_input(
        gridded, gridded, 1, model_comm_group=group, dataset_name="data"
    )
    assert coords.shape == (total_nodes, 2)
    assert latent.shape == (local_nodes, 1 + 1 + 4 + 2)
    assert sizes == node_sizes


@pytest.mark.distributed
def test_transport_conditioning_uses_flat_node_shards_and_global_embedding_gradients(
    distributed_backend, distributed_world_size
) -> None:
    _run_distributed_test(
        _check_transport_conditioning,
        backend=distributed_backend,
        world_size=distributed_world_size,
    )
