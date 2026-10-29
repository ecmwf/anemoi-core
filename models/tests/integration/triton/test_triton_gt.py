# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Tuple

import pytest
import torch

from anemoi.models.layers.conv import GraphTransformerConv
from anemoi.models.triton.utils import edge_index_to_csc
from anemoi.models.triton.utils import is_triton_available

if is_triton_available():
    from anemoi.models.triton.gt import graph_transformer_attention
    from anemoi.models.triton.gt import graph_transformer_attention_conv


@pytest.fixture(autouse=True)
def setup_torch():
    """Skip when CUDA/Triton are unavailable and set up torch defaults for all tests."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not is_triton_available():
        pytest.skip("Triton not available")
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float32)
    yield


def build_bipartite_graph(n_src: int, n_dst: int) -> Tuple[torch.Tensor, int]:
    """Build random bipartite graph and return edge_index and number of edges."""
    edges = []
    for dst in range(n_dst):
        deg = torch.randint(0, n_src, (1,)).item()
        srcs = torch.randperm(n_src)[:deg]
        edges.extend([(src.item(), dst) for src in srcs])

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return edge_index, edge_index.shape[1]


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_src,n_dst,h,d",
    [
        (4, 10, 2, 4),
        (4, 10, 6, 4),  # tests num_heads != pow_of_2
        (4, 10, 2, 6),  # tests  num_channels != pow_of_2
        (4, 10, 6, 6),  # tests num_heads * num_channels != pow_of_2
    ],
)
def test_graph_transformer_forward(n_src: int, n_dst: int, h: int, d: int):
    """Test forward pass of Triton GT."""
    edge_index, m = build_bipartite_graph(n_src, n_dst)
    csc, perm, reverse = edge_index_to_csc(edge_index, num_nodes=(n_src, n_dst), reverse=True)

    query = torch.randn((n_dst, h, d), requires_grad=True)
    key = torch.randn((n_src, h, d), requires_grad=True)
    value = torch.randn((n_src, h, d), requires_grad=True)
    edge_attr = torch.randn((m, h, d), requires_grad=True)

    edge_attr_csc = edge_attr[perm]
    out_triton = graph_transformer_attention_conv(query, key, value, edge_attr_csc, csc, reverse)

    # Verify output shape
    assert out_triton.shape == (n_dst, h, d), f"Expected shape {(n_dst, h, d)}, got {out_triton.shape}"

    # Verify output is not NaN or Inf
    assert torch.isfinite(out_triton).all(), "Output contains NaN or Inf"


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_src,n_dst,h,d",
    [
        (4, 10, 2, 4),
    ],
)
def test_graph_transformer_backward(n_src: int, n_dst: int, h: int, d: int):
    """Test backward pass of Triton GT."""
    edge_index, m = build_bipartite_graph(n_src, n_dst)
    csc, perm, reverse = edge_index_to_csc(edge_index, num_nodes=(n_src, n_dst), reverse=True)

    query = torch.randn((n_dst, h, d), requires_grad=True)
    key = torch.randn((n_src, h, d), requires_grad=True)
    value = torch.randn((n_src, h, d), requires_grad=True)
    edge_attr = torch.randn((m, h, d), requires_grad=True)

    edge_attr_csc = edge_attr[perm]
    out_triton = graph_transformer_attention_conv(query, key, value, edge_attr_csc, csc, reverse)
    loss = out_triton.pow(2).sum()
    loss.backward()

    # Verify gradients exist and are not NaN
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert key.grad is not None and torch.isfinite(key.grad).all()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert edge_attr.grad is not None and torch.isfinite(edge_attr.grad).all()


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_src,n_dst,h,d",
    [
        (4, 10, 2, 4),
        (4, 10, 6, 4),
        (4, 10, 2, 6),
        (4, 10, 6, 6),
    ],
)
def test_graph_transformer_vs_reference_forward(n_src: int, n_dst: int, h: int, d: int):
    """Test that triton Triton GT matches reference implementation."""
    edge_index, m = build_bipartite_graph(n_src, n_dst)
    csc, perm, reverse = edge_index_to_csc(edge_index, num_nodes=(n_src, n_dst), reverse=True)

    # Custom implementation
    query = torch.randn((n_dst, h, d), requires_grad=True)
    key = torch.randn((n_src, h, d), requires_grad=True)
    value = torch.randn((n_src, h, d), requires_grad=True)
    edge_attr = torch.randn((m, h, d), requires_grad=True)

    edge_attr_csc = edge_attr[perm]
    out_triton = graph_transformer_attention_conv(query, key, value, edge_attr_csc, csc, reverse)

    # Reference pyg implementation
    gt_ref = GraphTransformerConv(out_channels=d)
    out_ref = gt_ref.forward(query, key, value, edge_attr, edge_index)

    tolerance = 1e-4
    torch.testing.assert_close(out_triton, out_ref, atol=tolerance, rtol=0)


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_src,n_dst,h,d",
    [
        (4, 10, 2, 4),
        (4, 10, 6, 4),
        (4, 10, 2, 6),
        (4, 10, 6, 6),
    ],
)
def test_graph_transformer_vs_reference_backward(n_src: int, n_dst: int, h: int, d: int):
    """Test that triton Triton GT matches reference implementation."""
    edge_index, m = build_bipartite_graph(n_src, n_dst)
    csc, perm, reverse = edge_index_to_csc(edge_index, num_nodes=(n_src, n_dst), reverse=True)

    # Custom implementation
    query = torch.randn((n_dst, h, d), requires_grad=True)
    key = torch.randn((n_src, h, d), requires_grad=True)
    value = torch.randn((n_src, h, d), requires_grad=True)
    edge_attr = torch.randn((m, h, d), requires_grad=True)

    edge_attr_csc = edge_attr[perm]
    out_triton = graph_transformer_attention_conv(query, key, value, edge_attr_csc, csc, reverse)
    loss_triton = out_triton.pow(2).sum()
    loss_triton.backward()
    grads_triton = (query.grad.clone(), key.grad.clone(), value.grad.clone(), edge_attr.grad.clone())

    query.grad.zero_()
    key.grad.zero_()
    value.grad.zero_()
    edge_attr.grad.zero_()

    # Reference pyg implementation
    gt_ref = GraphTransformerConv(out_channels=d)
    out_ref = gt_ref.forward(query, key, value, edge_attr, edge_index)
    loss_ref = out_ref.pow(2).sum()
    loss_ref.backward()
    grads_ref = (query.grad.clone(), key.grad.clone(), value.grad.clone(), edge_attr.grad.clone())

    # Compare outputs and gradients
    tolerance = 1e-4
    torch.testing.assert_close(out_triton, out_ref, atol=tolerance, rtol=0)
    torch.testing.assert_close(grads_triton[0], grads_ref[0], atol=tolerance, rtol=0)  # queries
    torch.testing.assert_close(grads_triton[1], grads_ref[1], atol=tolerance, rtol=0)  # keys
    torch.testing.assert_close(grads_triton[2], grads_ref[2], atol=tolerance, rtol=0)  # values
    torch.testing.assert_close(grads_triton[3], grads_ref[3], atol=tolerance, rtol=0)  # edges


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_src,n_dst,h,d",
    [
        (4, 10, 2, 4),
        (4, 10, 6, 4),
        (4, 10, 2, 6),
        (4, 10, 6, 6),
    ],
)
def test_graph_transformer_attention_opcheck(n_src: int, n_dst: int, h: int, d: int):
    """Check the ``graph_transformer_attention`` custom op is registered correctly.

    ``torch.library.opcheck`` validates the schema, fake/meta implementation and
    autograd registration of the operator.
    """
    edge_index, m = build_bipartite_graph(n_src, n_dst)
    (row, colptr), perm, (rowptr, edge_ids, edge_dst) = edge_index_to_csc(
        edge_index, num_nodes=(n_src, n_dst), reverse=True
    )

    query = torch.randn((n_dst, h, d), requires_grad=True)
    key = torch.randn((n_src, h, d), requires_grad=True)
    value = torch.randn((n_src, h, d), requires_grad=True)
    edge_attr_csc = torch.randn((m, h, d), requires_grad=True)

    torch.library.opcheck(
        graph_transformer_attention,
        (query, key, value, edge_attr_csc, row, colptr, rowptr, edge_ids, edge_dst),
    )


#####################################################################
# int32 offset overflow                                             #
#####################################################################
# The kernels calculate memory offsets as ``index * H * C``. If the index is int32, this product
# overflows when it is ``2**31`` or more. The kernel then reads incorrect memory. The result is
# an illegal memory access or an incorrect output without an error.
#
# Each test runs the kernel on a large graph. Then it runs the kernel again on a small graph that
# contains only the last destination nodes and their edges. The two outputs must be equal, because
# the kernel calculates each destination node independently.

OVERFLOW_INT32_LIMIT = 2**31
OVERFLOW_NUM_HEADS = 16
OVERFLOW_NUM_CHANNELS = 64  # 16 * 64 = 1024 attention channels, as in the o1280 encoder
OVERFLOW_NUM_SRC = 1024
OVERFLOW_NUM_TAIL = 100  # number of destination nodes that we compare with the small graph
OVERFLOW_DTYPE = torch.bfloat16


def build_dst_sorted_graph(n_dst: int, degree: int, index_dtype: torch.dtype) -> torch.Tensor:
    """Build a dst-sorted bipartite edge_index where each destination node has ``degree`` sources."""
    dst = torch.arange(n_dst, dtype=torch.int64).repeat_interleave(degree)
    src = torch.arange(n_dst * degree, dtype=torch.int64) % OVERFLOW_NUM_SRC
    return torch.stack([src, dst]).to(index_dtype)


def run_gt_dst_sorted(query, key, value, edge_attr, edge_index):
    """Run the Triton GT on a dst-sorted edge_index."""
    n_src, n_dst = key.shape[0], query.shape[0]
    csc, _, reverse = edge_index_to_csc(edge_index, num_nodes=(n_src, n_dst), reverse=True, edges_are_dst_sorted=True)
    return graph_transformer_attention_conv(query, key, value, edge_attr, csc, reverse)


def check_tail_matches_subgraph(n_dst: int, degree: int, index_dtype: torch.dtype) -> None:
    """Compare the output and gradients of the last destination nodes with a small graph."""
    if torch.cuda.get_device_properties(0).total_memory < 40 * 2**30:
        pytest.skip("Test needs a GPU with 40 GB of memory or more")
    torch.manual_seed(0)
    num_edges = n_dst * degree

    edge_index = build_dst_sorted_graph(n_dst, degree, index_dtype)
    query = torch.randn((n_dst, OVERFLOW_NUM_HEADS, OVERFLOW_NUM_CHANNELS), dtype=OVERFLOW_DTYPE, requires_grad=True)
    key = torch.randn(
        (OVERFLOW_NUM_SRC, OVERFLOW_NUM_HEADS, OVERFLOW_NUM_CHANNELS), dtype=OVERFLOW_DTYPE, requires_grad=True
    )
    value = torch.randn(
        (OVERFLOW_NUM_SRC, OVERFLOW_NUM_HEADS, OVERFLOW_NUM_CHANNELS), dtype=OVERFLOW_DTYPE, requires_grad=True
    )
    edge_attr = torch.randn(
        (num_edges, OVERFLOW_NUM_HEADS, OVERFLOW_NUM_CHANNELS), dtype=OVERFLOW_DTYPE, requires_grad=True
    )

    out = run_gt_dst_sorted(query, key, value, edge_attr, edge_index)
    out_tail = out[-OVERFLOW_NUM_TAIL:].float()
    out_tail.sum().backward()
    grad_q_tail = query.grad[-OVERFLOW_NUM_TAIL:].float()
    grad_e_tail = edge_attr.grad[-OVERFLOW_NUM_TAIL * degree :].float()
    del out, edge_index
    edge_attr.grad = None
    query.grad = None

    # Small graph: the same last destination nodes, the same edges and the same source nodes.
    query_sub = query[-OVERFLOW_NUM_TAIL:].detach().clone().requires_grad_(True)
    edge_attr_sub = edge_attr[-OVERFLOW_NUM_TAIL * degree :].detach().clone().requires_grad_(True)
    edge_index_sub = build_dst_sorted_graph(n_dst, degree, torch.int64)[:, -OVERFLOW_NUM_TAIL * degree :]
    edge_index_sub[1] -= n_dst - OVERFLOW_NUM_TAIL

    out_sub = run_gt_dst_sorted(query_sub, key, value, edge_attr_sub, edge_index_sub).float()
    out_sub.sum().backward()

    assert torch.isfinite(out_tail).all(), "Output contains NaN or Inf"
    torch.testing.assert_close(out_tail, out_sub, atol=0, rtol=0, msg="Forward output is incorrect")
    torch.testing.assert_close(grad_q_tail, query_sub.grad.float(), atol=0, rtol=0, msg="Query gradient is incorrect")
    torch.testing.assert_close(
        grad_e_tail, edge_attr_sub.grad.float(), atol=0, rtol=0, msg="Edge gradient is incorrect"
    )
    torch.cuda.empty_cache()


@pytest.mark.slow
@pytest.mark.parametrize("index_dtype", [torch.int64, torch.int32], ids=["int64", "int32"])
def test_gt_edge_offset_overflow(index_dtype: torch.dtype):
    """Test a graph where ``num_edges * H * C`` is more than ``2**31``.

    The number of destination nodes is small enough that ``n_dst * H * C`` does not overflow.
    """
    degree = 4
    num_edges = OVERFLOW_INT32_LIMIT // (OVERFLOW_NUM_HEADS * OVERFLOW_NUM_CHANNELS) + 100_000
    n_dst = num_edges // degree
    assert n_dst * OVERFLOW_NUM_HEADS * OVERFLOW_NUM_CHANNELS < OVERFLOW_INT32_LIMIT
    assert n_dst * degree * OVERFLOW_NUM_HEADS * OVERFLOW_NUM_CHANNELS >= OVERFLOW_INT32_LIMIT

    check_tail_matches_subgraph(n_dst, degree, index_dtype)


@pytest.mark.slow
@pytest.mark.parametrize("index_dtype", [torch.int64, torch.int32], ids=["int64", "int32"])
def test_gt_dst_offset_overflow(index_dtype: torch.dtype):
    """Test a graph where ``n_dst * H * C`` is more than ``2**31``.

    The program id is always int32, thus this overflow occurs for all index dtypes.
    """
    degree = 1
    n_dst = OVERFLOW_INT32_LIMIT // (OVERFLOW_NUM_HEADS * OVERFLOW_NUM_CHANNELS) + 100_000
    assert n_dst * OVERFLOW_NUM_HEADS * OVERFLOW_NUM_CHANNELS >= OVERFLOW_INT32_LIMIT

    check_tail_matches_subgraph(n_dst, degree, index_dtype)
