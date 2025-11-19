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
from anemoi.models.triton.gt import GraphTransformerFunction
from anemoi.models.triton.utils import edge_index_to_csc


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


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute maximum absolute difference between two tensors."""
    return (a - b).abs().max().item()


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_src,n_dst,h,d",
    [
        (4, 10, 2, 4),
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
    ],
)
def test_graph_transformer_vs_reference(n_src: int, n_dst: int, h: int, d: int):
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
    output_diff = max_abs_diff(out_triton, out_ref)
    dq_diff = max_abs_diff(grads_triton[0], grads_ref[0])
    dk_diff = max_abs_diff(grads_triton[1], grads_ref[1])
    dv_diff = max_abs_diff(grads_triton[2], grads_ref[2])
    de_diff = max_abs_diff(grads_triton[3], grads_ref[3])

    tolerance = 1e-5
    assert output_diff < tolerance, f"Output diff {output_diff} exceeds tolerance {tolerance}"
    assert dq_diff < tolerance, f"Query gradient diff {dq_diff} exceeds tolerance {tolerance}"
    assert dk_diff < tolerance, f"Key gradient diff {dk_diff} exceeds tolerance {tolerance}"
    assert dv_diff < tolerance, f"Value gradient diff {dv_diff} exceeds tolerance {tolerance}"
    assert de_diff < tolerance, f"Edge attr gradient diff {de_diff} exceeds tolerance {tolerance}"
