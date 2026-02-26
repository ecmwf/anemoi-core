# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Optional
from typing import Tuple

import pytest
import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size

from anemoi.models.layers.conv import GraphTransformerConv
from anemoi.models.triton.utils import edge_index_to_csc
from anemoi.models.triton.utils import is_triton_available

if is_triton_available():
    from anemoi.models.triton.gt import GraphTransformer
    from anemoi.models.triton.gt import GraphTransformerFunction


@pytest.fixture(autouse=True)
def setup_torch():
    """Set up torch defaults for all tests."""
    device = "cuda"
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


class _GraphTransformerConvQKnorm(GraphTransformerConv):
    """Wrapper class around GraphTransformer which incorporates an RMSNorm.

    Intended for correctness testing purposes only.
    """

    def __init__(
        self,
        qk_norm: str = "none",
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.qk_norm = qk_norm
        if self.qk_norm == "rms_norm":  # rms norm
            self.q_norm = torch.nn.RMSNorm(self.out_channels)
            self.k_norm = torch.nn.RMSNorm(self.out_channels)
        elif self.qk_norm == "layer_norm":  # layer norm
            self.q_norm = torch.nn.LayerNorm(self.out_channels, bias=False)
            self.k_norm = torch.nn.LayerNorm(self.out_channels, bias=False)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        edge_attr: OptTensor,
        edge_index: Adj,
        size: Optional[Size] = None,
    ):
        if self.qk_norm != "none":
            query = self.q_norm(query)
            key = self.k_norm(key)

        return super().forward(query, key, value, edge_attr, edge_index, size=size)


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_src,n_dst,h,d,qk_norm",
    [
        (4, 10, 2, 4, 0),
        (4, 10, 6, 4, 0),  # tests num_heads != pow_of_2
        (4, 10, 2, 6, 0),  # tests  num_channels != pow_of_2
        (4, 10, 6, 6, 0),  # tests num_heads * num_channels != pow_of_2
        (4, 10, 6, 6, 1),  # tests fused layer normalisation
        (4, 10, 6, 6, 2),  # tests fused RMS normalisation
    ],
)
def test_graph_transformer_forward(n_src: int, n_dst: int, h: int, d: int, qk_norm: int):
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
    out_triton = GraphTransformerFunction.apply(query, key, value, edge_attr_csc, csc, reverse, qk_norm)

    # Verify output shape
    assert out_triton.shape == (n_dst, h, d), f"Expected shape {(n_dst, h, d)}, got {out_triton.shape}"

    # Verify output is not NaN or Inf
    assert torch.isfinite(out_triton).all(), "Output contains NaN or Inf"


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_src,n_dst,h,d,qk_norm",
    [
        (4, 10, 2, 4, 0),
        (4, 10, 6, 4, 0),  # tests num_heads != pow_of_2
        (4, 10, 2, 6, 0),  # tests  num_channels != pow_of_2
        (4, 10, 6, 6, 0),  # tests num_heads * num_channels != pow_of_2
        (4, 10, 6, 6, 1),  # tests fused layer normalisation
        (4, 10, 6, 6, 2),  # tests fused RMS normalisation
    ],
)
def test_graph_transformer_backward(n_src: int, n_dst: int, h: int, d: int, qk_norm: int):
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
    out_triton = GraphTransformerFunction.apply(
        query,
        key,
        value,
        edge_attr_csc,
        csc,
        reverse,
        qk_norm,
    )
    loss = out_triton.pow(2).sum()
    loss.backward()

    # Verify gradients exist and are not NaN
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert key.grad is not None and torch.isfinite(key.grad).all()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert edge_attr.grad is not None and torch.isfinite(edge_attr.grad).all()


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_src,n_dst,h,d,qk_norm",
    [(4, 10, H, C, qk_norm) for H in (2, 6) for C in (4, 6) for qk_norm in ("none", "rms_norm", "layer_norm")],
)
def test_graph_transformer_vs_reference_forward(n_src: int, n_dst: int, h: int, d: int, qk_norm: str):
    """Test that triton GraphTransformerFunction matches reference implementation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    edge_index, m = build_bipartite_graph(n_src, n_dst)

    # Custom implementation
    query_triton = torch.randn((n_dst, h, d), requires_grad=True)
    key_triton = torch.randn((n_src, h, d), requires_grad=True)
    value = torch.randn((n_src, h, d), requires_grad=True)
    edge_attr = torch.randn((m, h, d), requires_grad=True)

    query_ref = torch.clone(query_triton)
    key_ref = torch.clone(key_triton)

    gt_triton = GraphTransformer(d, qk_norm)
    # Reference pyg implementation
    gt_ref = _GraphTransformerConvQKnorm(out_channels=d, qk_norm=qk_norm)

    size = (n_src, n_dst)
    csc, perm, reverse = edge_index_to_csc(edge_index, num_nodes=size, reverse=True)
    edge_attr_csc = edge_attr[perm]

    out_triton = gt_triton(
        query_triton,
        key_triton,
        value,
        edge_attr_csc,
        csc,
        reverse,
    )

    out_ref = gt_ref.forward(query_ref, key_ref, value, edge_attr, edge_index, size=(value.size(0), query_ref.size(0)))

    tolerance = 1e-4
    torch.testing.assert_close(out_triton, out_ref, atol=tolerance, rtol=0)


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_src,n_dst,h,d,qk_norm",
    [(4, 10, H, C, qk_norm) for H in (2, 6) for C in (4, 6) for qk_norm in ("none", "rms_norm", "layer_norm")],
)
def test_graph_transformer_vs_reference_backward(n_src: int, n_dst: int, h: int, d: int, qk_norm: str):
    """Test that triton GraphTransformerFunction matches reference implementation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    edge_index, m = build_bipartite_graph(n_src, n_dst)

    # Custom implementation
    query = torch.randn((n_dst, h, d), requires_grad=True)
    key = torch.randn((n_src, h, d), requires_grad=True)
    value = torch.randn((n_src, h, d), requires_grad=True)
    edge_attr = torch.randn((m, h, d), requires_grad=True)

    gt_triton = GraphTransformer(d, qk_norm)
    # Reference pyg implementation
    gt_ref = _GraphTransformerConvQKnorm(out_channels=d, qk_norm=qk_norm)

    size = (n_src, n_dst)
    csc, perm, reverse = edge_index_to_csc(edge_index, num_nodes=size, reverse=True)
    edge_attr_csc = edge_attr[perm]

    out_triton = gt_triton(
        query,
        key,
        value,
        edge_attr_csc,
        csc,
        reverse,
    )

    loss_triton = out_triton.pow(2).sum()
    loss_triton.backward()
    grads_triton = (query.grad.clone(), key.grad.clone(), value.grad.clone(), edge_attr.grad.clone())

    query.grad.zero_()
    key.grad.zero_()
    value.grad.zero_()
    edge_attr.grad.zero_()

    out_ref = gt_ref.forward(query, key, value, edge_attr, edge_index, size=(value.size(0), query.size(0)))
    loss_ref = out_ref.pow(2).sum()
    loss_ref.backward()
    grads_ref = (query.grad.clone(), key.grad.clone(), value.grad.clone(), edge_attr.grad.clone())
    # Compare outputs and gradients
    tolerance = 1e-4
    torch.testing.assert_close(out_triton, out_ref, atol=tolerance, rtol=0)
    torch.testing.assert_close(grads_triton[1], grads_ref[1], atol=tolerance, rtol=0)  # keys
    torch.testing.assert_close(grads_triton[2], grads_ref[2], atol=tolerance, rtol=0)  # values
    torch.testing.assert_close(grads_triton[3], grads_ref[3], atol=tolerance, rtol=0)  # edges
