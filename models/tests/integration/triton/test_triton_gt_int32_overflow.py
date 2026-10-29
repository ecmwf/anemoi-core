# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Integration tests for int32 offset overflow in the Triton GraphTransformer kernels.

The kernels calculate memory offsets as ``index * H * C``. If the index is int32, this product
overflows when it is ``2**31`` or more. The kernel then reads incorrect memory. The result is
an illegal memory access or an incorrect output without an error.

Each test runs the kernel on a large graph. Then it runs the kernel again on a small graph that
contains only the last destination nodes and their edges. The two outputs must be equal, because
the kernel calculates each destination node independently.
"""

import pytest
import torch

from anemoi.models.triton.utils import edge_index_to_csc
from anemoi.models.triton.utils import is_triton_available

if is_triton_available():
    from anemoi.models.triton.gt import graph_transformer_attention_conv

INT32_LIMIT = 2**31
NUM_HEADS = 16
NUM_CHANNELS = 64  # NUM_HEADS * NUM_CHANNELS = 1024 attention channels, as in the o1280 encoder
NUM_SRC = 1024
NUM_TAIL = 100  # number of destination nodes that we compare with the small graph
DTYPE = torch.bfloat16


@pytest.fixture(autouse=True)
def setup_torch():
    """Skip when CUDA/Triton are unavailable or the GPU memory is too small."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not is_triton_available():
        pytest.skip("Triton not available")
    if torch.cuda.get_device_properties(0).total_memory < 40 * 2**30:
        pytest.skip("Test needs a GPU with 40 GB of memory or more")
    torch.set_default_device("cuda")
    yield
    torch.cuda.empty_cache()


def build_graph(n_dst: int, degree: int, index_dtype: torch.dtype) -> torch.Tensor:
    """Build a dst-sorted bipartite edge_index where each destination node has ``degree`` sources."""
    dst = torch.arange(n_dst, dtype=torch.int64).repeat_interleave(degree)
    src = torch.arange(n_dst * degree, dtype=torch.int64) % NUM_SRC
    return torch.stack([src, dst]).to(index_dtype)


def run_gt(query, key, value, edge_attr, edge_index):
    """Run the Triton GT on a dst-sorted edge_index."""
    n_src, n_dst = key.shape[0], query.shape[0]
    csc, _, reverse = edge_index_to_csc(edge_index, num_nodes=(n_src, n_dst), reverse=True, edges_are_dst_sorted=True)
    return graph_transformer_attention_conv(query, key, value, edge_attr, csc, reverse)


def check_tail_matches_subgraph(n_dst: int, degree: int, index_dtype: torch.dtype) -> None:
    """Compare the output and gradients of the last destination nodes with a small graph."""
    torch.manual_seed(0)
    num_edges = n_dst * degree

    edge_index = build_graph(n_dst, degree, index_dtype)
    query = torch.randn((n_dst, NUM_HEADS, NUM_CHANNELS), dtype=DTYPE, requires_grad=True)
    key = torch.randn((NUM_SRC, NUM_HEADS, NUM_CHANNELS), dtype=DTYPE, requires_grad=True)
    value = torch.randn((NUM_SRC, NUM_HEADS, NUM_CHANNELS), dtype=DTYPE, requires_grad=True)
    edge_attr = torch.randn((num_edges, NUM_HEADS, NUM_CHANNELS), dtype=DTYPE, requires_grad=True)

    out = run_gt(query, key, value, edge_attr, edge_index)
    out_tail = out[-NUM_TAIL:].float()
    out_tail.sum().backward()
    grad_q_tail = query.grad[-NUM_TAIL:].float()
    grad_e_tail = edge_attr.grad[-NUM_TAIL * degree :].float()
    del out, edge_index
    edge_attr.grad = None
    query.grad = None

    # Small graph: the same last destination nodes, the same edges and the same source nodes.
    query_sub = query[-NUM_TAIL:].detach().clone().requires_grad_(True)
    edge_attr_sub = edge_attr[-NUM_TAIL * degree :].detach().clone().requires_grad_(True)
    edge_index_sub = build_graph(n_dst, degree, torch.int64)[:, -NUM_TAIL * degree :]
    edge_index_sub[1] -= n_dst - NUM_TAIL

    out_sub = run_gt(query_sub, key, value, edge_attr_sub, edge_index_sub).float()
    out_sub.sum().backward()

    assert torch.isfinite(out_tail).all(), "Output contains NaN or Inf"
    torch.testing.assert_close(out_tail, out_sub, atol=0, rtol=0, msg="Forward output is incorrect")
    torch.testing.assert_close(grad_q_tail, query_sub.grad.float(), atol=0, rtol=0, msg="Query gradient is incorrect")
    torch.testing.assert_close(grad_e_tail, edge_attr_sub.grad.float(), atol=0, rtol=0, msg="Edge gradient is incorrect")


@pytest.mark.slow
@pytest.mark.parametrize("index_dtype", [torch.int64, torch.int32], ids=["int64", "int32"])
def test_gt_edge_offset_overflow(index_dtype: torch.dtype):
    """Test a graph where ``num_edges * H * C`` is more than ``2**31``.

    The number of destination nodes is small enough that ``n_dst * H * C`` does not overflow.
    """
    degree = 4
    num_edges = INT32_LIMIT // (NUM_HEADS * NUM_CHANNELS) + 100_000
    n_dst = num_edges // degree
    assert n_dst * NUM_HEADS * NUM_CHANNELS < INT32_LIMIT
    assert n_dst * degree * NUM_HEADS * NUM_CHANNELS >= INT32_LIMIT

    check_tail_matches_subgraph(n_dst, degree, index_dtype)


@pytest.mark.slow
@pytest.mark.parametrize("index_dtype", [torch.int64, torch.int32], ids=["int64", "int32"])
def test_gt_dst_offset_overflow(index_dtype: torch.dtype):
    """Test a graph where ``n_dst * H * C`` is more than ``2**31``.

    The program id is always int32, thus this overflow occurs for all index dtypes.
    """
    degree = 1
    n_dst = INT32_LIMIT // (NUM_HEADS * NUM_CHANNELS) + 100_000
    assert n_dst * NUM_HEADS * NUM_CHANNELS >= INT32_LIMIT

    check_tail_matches_subgraph(n_dst, degree, index_dtype)
