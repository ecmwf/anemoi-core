# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.models.layers.sparse_projector import SparseProjector


def _make_provider(
    n_rows: int, n_cols: int, nnz: int, seed: int = 0, backend: str = "torch"
) -> ProjectionGraphProvider:
    """A projection from ``n_cols`` source nodes to ``n_rows`` target nodes with ``nnz`` random weights."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    flat_indices = torch.randperm(n_rows * n_cols, generator=rng)[:nnz]
    graph = HeteroData()
    graph["src"].num_nodes = n_cols
    graph["dst"].num_nodes = n_rows
    graph[("src", "to", "dst")].edge_index = torch.stack([flat_indices % n_cols, flat_indices // n_cols])
    graph[("src", "to", "dst")].weight = torch.rand(nnz, generator=rng)
    return ProjectionGraphProvider(
        graph=graph, edges_name=("src", "to", "dst"), edge_weight_attribute="weight", backend=backend
    )


@pytest.fixture
def projector() -> SparseProjector:
    return SparseProjector(autocast=False)


@pytest.fixture
def provider() -> ProjectionGraphProvider:
    return _make_provider(n_rows=8, n_cols=16, nnz=32, seed=0)


def test_forward_matches_dense_matmul(projector: SparseProjector, provider: ProjectionGraphProvider) -> None:
    """forward result matches the equivalent dense matrix multiplication."""
    x = torch.rand(2, 16, 4)
    out = projector(x, provider)
    dense = provider.get_edges().to_dense()
    expected = torch.einsum("mn,bnc->bmc", dense, x)
    assert torch.allclose(out, expected, atol=1e-5)


def test_forward_handles_arbitrary_leading_dims(projector: SparseProjector, provider: ProjectionGraphProvider) -> None:
    """forward correctly handles inputs with extra leading dimensions."""
    # Shape: [batch, timesteps, nodes, vars]
    x = torch.rand(2, 3, 16, 5)
    out = projector(x, provider)
    assert out.shape == (2, 3, 8, 5)
    # Verify values match per-sample forward
    for b in range(2):
        for t in range(3):
            expected = projector(x[b, t : t + 1], provider)
            assert torch.allclose(out[b, t], expected[0], atol=1e-6)


def test_forward_uses_configured_num_chunks(provider: ProjectionGraphProvider) -> None:
    x = torch.rand(5, 16, 4)
    chunked = SparseProjector(autocast=False, num_chunks=3)(x, provider)
    unchunked = SparseProjector(autocast=False, num_chunks=1)(x, provider)
    torch.testing.assert_close(chunked, unchunked)


def test_forward_rejects_wrong_number_of_source_nodes(
    projector: SparseProjector, provider: ProjectionGraphProvider
) -> None:
    with pytest.raises(ValueError, match="16 source nodes"):
        projector(torch.rand(2, 15, 4), provider)


def test_triton_backend_needs_a_gpu() -> None:
    provider = _make_provider(n_rows=8, n_cols=16, nnz=32, backend="triton")
    with pytest.raises(ValueError, match="GPU"):
        SparseProjector()(torch.rand(2, 16, 4), provider)


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="backend"):
        _make_provider(n_rows=8, n_cols=16, nnz=32, backend="cuda")


def test_forward_of_a_field_with_zero_channels(projector: SparseProjector, provider: ProjectionGraphProvider) -> None:
    x = torch.rand(2, 3, 16, 0, requires_grad=True)

    out = projector(x, provider)
    out.sum().backward()

    assert out.shape == (2, 3, 8, 0)
    assert x.grad.shape == x.shape
