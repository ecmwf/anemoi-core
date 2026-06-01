# (C) Copyright 2025 Anemoi contributors.
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
    from anemoi.models.triton.gt import GraphTransformerFunction


@pytest.fixture(autouse=True)
def setup_torch():
    """Set up torch defaults for all tests."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
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
    """Test forward pass of GraphTransformerFunction."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    edge_index, m = build_bipartite_graph(n_src, n_dst)
    csc, perm, reverse = edge_index_to_csc(edge_index, num_nodes=(n_src, n_dst), reverse=True)

    query = torch.randn((n_dst, h, d), requires_grad=True)
    key = torch.randn((n_src, h, d), requires_grad=True)
    value = torch.randn((n_src, h, d), requires_grad=True)
    edge_attr = torch.randn((m, h, d), requires_grad=True)

    edge_attr_csc = edge_attr[perm]
    out_triton = GraphTransformerFunction.apply(query, key, value, edge_attr_csc, csc, reverse)

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
    """Test backward pass of GraphTransformerFunction."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    edge_index, m = build_bipartite_graph(n_src, n_dst)
    csc, perm, reverse = edge_index_to_csc(edge_index, num_nodes=(n_src, n_dst), reverse=True)

    query = torch.randn((n_dst, h, d), requires_grad=True)
    key = torch.randn((n_src, h, d), requires_grad=True)
    value = torch.randn((n_src, h, d), requires_grad=True)
    edge_attr = torch.randn((m, h, d), requires_grad=True)

    edge_attr_csc = edge_attr[perm]
    out_triton = GraphTransformerFunction.apply(query, key, value, edge_attr_csc, csc, reverse)
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
    """Test that triton GraphTransformerFunction matches reference implementation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    edge_index, m = build_bipartite_graph(n_src, n_dst)
    csc, perm, reverse = edge_index_to_csc(edge_index, num_nodes=(n_src, n_dst), reverse=True)

    # Custom implementation
    query = torch.randn((n_dst, h, d), requires_grad=True)
    key = torch.randn((n_src, h, d), requires_grad=True)
    value = torch.randn((n_src, h, d), requires_grad=True)
    edge_attr = torch.randn((m, h, d), requires_grad=True)

    edge_attr_csc = edge_attr[perm]
    out_triton = GraphTransformerFunction.apply(query, key, value, edge_attr_csc, csc, reverse)

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
    """Test that triton GraphTransformerFunction matches reference implementation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    edge_index, m = build_bipartite_graph(n_src, n_dst)
    csc, perm, reverse = edge_index_to_csc(edge_index, num_nodes=(n_src, n_dst), reverse=True)

    # Custom implementation
    query = torch.randn((n_dst, h, d), requires_grad=True)
    key = torch.randn((n_src, h, d), requires_grad=True)
    value = torch.randn((n_src, h, d), requires_grad=True)
    edge_attr = torch.randn((m, h, d), requires_grad=True)

    edge_attr_csc = edge_attr[perm]
    out_triton = GraphTransformerFunction.apply(query, key, value, edge_attr_csc, csc, reverse)
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


def _build_star_graph(n_src, device="cuda"):
    """Build a simple star graph: n_src source nodes all pointing to 1 destination.

    Returns CSC and reverse (source-grouped) representations.

    Graph: src_0 -> dst, src_1 -> dst, ..., src_{n-1} -> dst
    The destination node is node 0 in the dst indexing.
    Source nodes are indexed 0..n_src-1.
    """
    n_dst = 1
    num_edges = n_src

    # CSC format (edges grouped by destination):
    # row[i] = source node of edge i
    # colptr[j] = start of edges for destination j
    row = torch.arange(n_src, device=device, dtype=torch.int64)
    colptr = torch.tensor([0, num_edges], device=device, dtype=torch.int64)

    # Reverse / source-grouped format:
    # rowptr[i] = start of edges for source i (each source has exactly 1 edge)
    # edge_ids[i] = original edge index for the i-th source-grouped edge
    # edge_dst[i] = destination node of edge i (all go to node 0)
    rowptr = torch.arange(n_src + 1, device=device, dtype=torch.int64)
    edge_ids = torch.arange(num_edges, device=device, dtype=torch.int64)
    edge_dst = torch.zeros(num_edges, device=device, dtype=torch.int64)

    csc = (row, colptr)
    reverse = (rowptr, edge_ids, edge_dst)
    return n_dst, n_src, num_edges, csc, reverse


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("scale", [1e3, 1e4])
def test_sharp_softmax_uniform_message_grads_zero(dtype, scale):
    """When all neighbors send the same message, dq and dk must be ~zero.

    We make one neighbor have a much larger attention score than the others
    (sharp softmax, scale in {1e3, 1e4}), but set v + e to be the same
    constant for every edge.  Since the output equals that constant regardless
    of the attention weights, the gradients w.r.t. q and k should vanish.
    """
    device = "cuda"
    H, C = 4, 16  # heads, channels
    n_neighbors = 8

    n_dst, n_src, num_edges, csc, reverse = _build_star_graph(n_neighbors, device)

    torch.manual_seed(42)

    # q is set so that dot(q, k_0 + e_0) >> dot(q, k_i + e_i) for i != 0
    # We achieve this by making k[0] aligned with q and scaled up.
    q = torch.randn(n_dst, H, C, device=device, dtype=dtype)

    # All edges deliver the SAME message: v[i] + e[i] = constant
    # Choose a fixed message vector.
    common_message = torch.randn(1, H, C, device=device, dtype=dtype)

    # Set v to the common message, e to zero => v + e = common_message for all
    v = common_message.expand(n_src, H, C).contiguous()
    e = torch.zeros(num_edges, H, C, device=device, dtype=dtype)

    # Make k such that one neighbor has a much larger score:
    # k[0] = scale * q (huge alignment), k[i!=0] = small random
    k = torch.randn(n_src, H, C, device=device, dtype=dtype) * 0.01
    k[0] = q[0] * scale  # This makes qk[0] ≈ scale * ||q||^2 >> others

    # Require gradients on q, k, v, e
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    e.requires_grad_(True)

    # Forward
    out = GraphTransformerFunction.apply(q, k, v, e, csc, reverse)

    # The output should be approximately the common message (since all v+e are the same)
    torch.testing.assert_close(out.float(), common_message.expand_as(out).float(), atol=1e-4, rtol=1e-4)

    # Backward with a random gradient
    d_out = torch.randn_like(out)
    out.backward(d_out)

    # dq and dk should be zero because the output doesn't depend on q or k
    # when all messages are identical.
    dq = q.grad
    dk = dk = k.grad

    # Use a relative tolerance based on the magnitude of dv (which IS nonzero)
    dv = v.grad
    dv_norm = dv.float().norm().item()

    # dq and dk norms should be negligible compared to dv
    dq_norm = dq.float().norm().item()
    dk_norm = dk.float().norm().item()

    # The relative error: dq_norm / dv_norm should be very small
    rel_tol = 1e-3  # generous tolerance - the bug produces errors >> this

    assert dq_norm / (dv_norm + 1e-12) < rel_tol, (
        f"dq should be ~zero when all messages are identical, but "
        f"|dq|/|dv| = {dq_norm/dv_norm:.6e} (scale={scale}, dtype={dtype}). "
        f"This indicates catastrophic cancellation in the backward pass."
    )

    assert dk_norm / (dv_norm + 1e-12) < rel_tol, (
        f"dk should be ~zero when all messages are identical, but "
        f"|dk|/|dv| = {dk_norm/dv_norm:.6e} (scale={scale}, dtype={dtype}). "
        f"This indicates catastrophic cancellation in the backward pass."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sharp_softmax_larger_graph(dtype):
    """Same principle as above but with multiple destination nodes.

    Each destination has several incoming edges, all delivering the same
    per-destination message. The gradients dq and dk should still be zero.
    """
    device = "cuda"
    H, C = 4, 16
    n_dst = 8
    n_src = 8  # bipartite: each src connects to each dst
    scale = 1e4

    # Build a complete bipartite graph (every src -> every dst)
    # CSC: for each dst, list all src nodes as neighbors
    num_edges = n_dst * n_src
    row_list = []
    colptr_list = [0]
    for d in range(n_dst):
        for s in range(n_src):
            row_list.append(s)
        colptr_list.append(len(row_list))

    row = torch.tensor(row_list, device=device, dtype=torch.int64)
    colptr = torch.tensor(colptr_list, device=device, dtype=torch.int64)

    # Reverse: for each src, list all edges (in original edge order)
    # Edge ordering in CSC: edge for (dst=d, src=s) is at index d*n_src + s
    rowptr_list = [0]
    edge_ids_list = []
    edge_dst_list = [0] * num_edges  # will fill from CSC
    for s in range(n_src):
        for d in range(n_dst):
            edge_id = d * n_src + s
            edge_ids_list.append(edge_id)
        rowptr_list.append(len(edge_ids_list))

    # edge_dst maps original edge id -> dst node
    for d in range(n_dst):
        for s in range(n_src):
            edge_dst_list[d * n_src + s] = d

    rowptr = torch.tensor(rowptr_list, device=device, dtype=torch.int64)
    edge_ids = torch.tensor(edge_ids_list, device=device, dtype=torch.int64)
    edge_dst = torch.tensor(edge_dst_list, device=device, dtype=torch.int64)

    csc = (row, colptr)
    reverse = (rowptr, edge_ids, edge_dst)

    torch.manual_seed(123)

    q = torch.randn(n_dst, H, C, device=device, dtype=dtype)
    # All messages identical: pick a common message per destination
    # To ensure v[s] + e[d*n_src + s] = constant for all s (for each dst d),
    # set e = 0 and v all the same
    common_message = torch.randn(1, H, C, device=device, dtype=dtype)
    v = common_message.expand(n_src, H, C).contiguous()
    e = torch.zeros(num_edges, H, C, device=device, dtype=dtype)

    # Make k[0] strongly aligned with all q, others small
    k = torch.randn(n_src, H, C, device=device, dtype=dtype) * 0.01
    k[0] = scale * torch.randn(1, H, C, device=device, dtype=dtype)

    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    e.requires_grad_(True)

    out = GraphTransformerFunction.apply(q, k, v, e, csc, reverse)
    d_out = torch.randn_like(out)
    out.backward(d_out)

    dq_norm = q.grad.float().norm().item()
    dk_norm = k.grad.float().norm().item()
    dv_norm = v.grad.float().norm().item()

    rel_tol = 1e-3

    assert dq_norm / (dv_norm + 1e-12) < rel_tol, (
        f"dq should be ~zero (uniform messages), but "
        f"|dq|/|dv| = {dq_norm/dv_norm:.6e} (dtype={dtype}). "
        f"Catastrophic cancellation in backward."
    )

    assert dk_norm / (dv_norm + 1e-12) < rel_tol, (
        f"dk should be ~zero (uniform messages), but "
        f"|dk|/|dv| = {dk_norm/dv_norm:.6e} (dtype={dtype}). "
        f"Catastrophic cancellation in backward."
    )
