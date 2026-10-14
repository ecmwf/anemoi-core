# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The Triton and PyG graph-attention backends must be interchangeable at the block level.

The layer/block tests elsewhere only construct the "pyg" backend on CPU, so the Triton wiring in
GraphTransformerBlock (CSC conversion, edge permutation, edges_are_dst_sorted, head sharding, inference chunking)
is otherwise untested. Here identical blocks (same weights) are run with both backends in fp32 and outputs plus
all parameter and input gradients are compared.
"""

import pytest
import torch

import anemoi.models.layers.block as block_module
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.block import GraphTransformerMapperBlock
from anemoi.models.layers.block import GraphTransformerProcessorBlock
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.models.triton.utils import is_triton_available

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not is_triton_available(), reason="Triton/CUDA not available"),
]

DEVICE = "cuda"
ATOL = RTOL = 1e-4
IN_CHANNELS, HIDDEN_DIM, EDGE_DIM, NUM_HEADS = 32, 64, 5, 4


def _random_graph(n_src, n_dst, max_deg, gen, dst_sorted=True):
    edges = []
    for dst in range(n_dst):
        deg = int(torch.randint(0, max_deg + 1, (1,), generator=gen, device="cpu"))  # includes isolated dst nodes
        edges += [(int(s), dst) for s in torch.randperm(n_src, generator=gen, device="cpu")[:deg]]
    edge_index = torch.tensor(edges, dtype=torch.long, device="cpu").t().contiguous()
    if not dst_sorted:
        edge_index = edge_index[:, torch.randperm(edge_index.shape[1], generator=gen, device="cpu")]
    return edge_index.to(DEVICE)


def _make_pair(cls, **kwargs):
    """The same block twice, once per backend, with identical weights."""
    blocks = {}
    for backend in ("pyg", "triton"):
        torch.manual_seed(0)
        blocks[backend] = cls(
            in_channels=IN_CHANNELS,
            hidden_dim=HIDDEN_DIM,
            out_channels=IN_CHANNELS,
            num_heads=NUM_HEADS,
            edge_dim=EDGE_DIM,
            layer_kernels=load_layer_kernels(),
            graph_attention_backend=backend,
            **kwargs,
        ).to(DEVICE)
    blocks["triton"].load_state_dict(blocks["pyg"].state_dict())
    return blocks


def _assert_same_forward_backward(blocks, make_inputs, call):
    results = {}
    for backend, blk in blocks.items():
        inputs = make_inputs()
        out = call(blk, inputs)
        outs = out if isinstance(out, tuple) else (out,)
        loss = sum((o * torch.linspace(-1, 1, o.numel(), device=DEVICE).view_as(o)).sum() for o in outs)
        loss.backward()
        leaf_grads = [t.grad for t in inputs if isinstance(t, torch.Tensor) and t.requires_grad]
        results[backend] = (
            outs,
            leaf_grads,
            {n: p.grad for n, p in blk.named_parameters()},
        )

    (o_p, g_p, pg_p), (o_t, g_t, pg_t) = results["pyg"], results["triton"]
    for a, b in zip(o_t, o_p):
        torch.testing.assert_close(a, b, atol=ATOL, rtol=RTOL)
    for a, b in zip(g_t, g_p):
        torch.testing.assert_close(a, b, atol=ATOL, rtol=RTOL)
    for name in pg_p:
        if pg_p[name] is None:
            assert pg_t[name] is None, name
            continue
        torch.testing.assert_close(pg_t[name], pg_p[name], atol=ATOL, rtol=RTOL, msg=lambda m: f"{name}: {m}")


@pytest.mark.parametrize("shard_strategy", ["edges", "heads"])
@pytest.mark.parametrize("qk_norm", [False, True])
@pytest.mark.parametrize("dst_sorted", [True, False])
def test_processor_block_triton_matches_pyg(shard_strategy, qk_norm, dst_sorted):
    gen = torch.Generator().manual_seed(1)
    n = 50
    edge_index = _random_graph(n, n, 8, gen, dst_sorted=dst_sorted)
    x0 = torch.randn(n, IN_CHANNELS, generator=gen, device="cpu").to(DEVICE)
    e0 = torch.randn(edge_index.shape[1], EDGE_DIM, generator=gen, device="cpu").to(DEVICE)
    blocks = _make_pair(
        GraphTransformerProcessorBlock,
        qk_norm=qk_norm,
        shard_strategy=shard_strategy,
        edge_pre_mlp=True,
    )

    def make_inputs():
        return x0.clone().requires_grad_(True), e0.clone().requires_grad_(True)

    def call(blk, inputs):
        x, e = inputs
        shard_info = GraphShardInfo(nodes=[n], edges=[edge_index.shape[1]])
        out, _ = blk(
            x,
            e,
            edge_index,
            shard_info,
            batch_size=1,
            size=n,
            edges_are_dst_sorted=dst_sorted,
        )
        return out

    _assert_same_forward_backward(blocks, make_inputs, call)


@pytest.mark.parametrize("shard_strategy", ["edges", "heads"])
@pytest.mark.parametrize("update_src_nodes", [False, True])
@pytest.mark.parametrize("dst_sorted", [True, False])
def test_mapper_block_triton_matches_pyg(shard_strategy, update_src_nodes, dst_sorted):
    gen = torch.Generator().manual_seed(2)
    n_src, n_dst = 60, 35
    edge_index = _random_graph(n_src, n_dst, 6, gen, dst_sorted=dst_sorted)
    xs0 = torch.randn(n_src, IN_CHANNELS, generator=gen, device="cpu").to(DEVICE)
    xd0 = torch.randn(n_dst, IN_CHANNELS, generator=gen, device="cpu").to(DEVICE)
    e0 = torch.randn(edge_index.shape[1], EDGE_DIM, generator=gen, device="cpu").to(DEVICE)
    blocks = _make_pair(
        GraphTransformerMapperBlock,
        qk_norm=True,
        shard_strategy=shard_strategy,
        update_src_nodes=update_src_nodes,
    )

    def make_inputs():
        return (
            xs0.clone().requires_grad_(True),
            xd0.clone().requires_grad_(True),
            e0.clone().requires_grad_(True),
        )

    def call(blk, inputs):
        xs, xd, e = inputs
        shard_info = BipartiteGraphShardInfo(src_nodes=[n_src], dst_nodes=[n_dst], edges=[edge_index.shape[1]])
        (out_src, out_dst), _ = blk(
            (xs, xd),
            e,
            edge_index,
            shard_info,
            batch_size=1,
            size=(n_src, n_dst),
            edges_are_dst_sorted=dst_sorted,
        )
        return (out_src, out_dst) if update_src_nodes else out_dst

    _assert_same_forward_backward(blocks, make_inputs, call)


@pytest.mark.parametrize("num_chunks", [2, 3, 7])
def test_processor_block_inference_chunking_triton(monkeypatch, num_chunks):
    """Inference chunking splits edges by dst and sums chunk outputs, which relies on the Triton kernel returning
    exact zeros for dst nodes without edges in a chunk."""
    gen = torch.Generator().manual_seed(3)
    n = 64
    edge_index = _random_graph(n, n, 8, gen)
    x = torch.randn(n, IN_CHANNELS, generator=gen, device="cpu").to(DEVICE)
    e = torch.randn(edge_index.shape[1], EDGE_DIM, generator=gen, device="cpu").to(DEVICE)
    blocks = _make_pair(GraphTransformerProcessorBlock, qk_norm=True)
    shard_info = GraphShardInfo(nodes=[n], edges=[edge_index.shape[1]])

    outs = {}
    with torch.no_grad():
        for backend, blk in blocks.items():
            blk.eval()
            for chunks in (1, num_chunks):
                monkeypatch.setattr(block_module, "NUM_CHUNKS_INFERENCE_PROCESSOR", chunks)
                outs[backend, chunks], _ = blk(x, e, edge_index, shard_info, batch_size=1, size=n)

    torch.testing.assert_close(outs["triton", num_chunks], outs["triton", 1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(outs["triton", num_chunks], outs["pyg", num_chunks], atol=ATOL, rtol=RTOL)
