# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from torch_geometric.data import HeteroData

from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.models.layers.graph_provider import use_torch_projections
from anemoi.models.layers.sparse_projector import SparseProjector


def test_projection_graph_provider_preserves_row_normalized_weights() -> None:
    graph = HeteroData()
    graph["src"].num_nodes = 3
    graph["dst"].num_nodes = 2

    edge_index = torch.tensor([[0, 1, 2, 0], [0, 0, 1, 1]])
    edge_weight = torch.tensor([0.25, 0.75, 0.6, 0.4])  # per-target sums: [1.0, 1.0]

    graph[("src", "to", "dst")].edge_index = edge_index
    graph[("src", "to", "dst")].gauss_weight = edge_weight

    provider = ProjectionGraphProvider(
        graph=graph,
        edges_name=("src", "to", "dst"),
        edge_weight_attribute="gauss_weight",
        row_normalize=False,
        backend="torch",
    )

    edges = provider.get_edges()
    assert edges.layout == torch.sparse_csr

    matrix = edges.to_dense()
    assert matrix.shape == (graph["dst"].num_nodes, graph["src"].num_nodes)

    row_sums = matrix.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_projection_graph_provider_accepts_int32_edge_index() -> None:
    graph = HeteroData()
    graph["src"].num_nodes = 3
    graph["dst"].num_nodes = 2

    # GraphCreator may yield int32 edge indices; provider should handle this.
    edge_index = torch.tensor([[0, 1, 2, 0], [0, 0, 1, 1]], dtype=torch.int32)
    edge_weight = torch.tensor([0.25, 0.75, 0.6, 0.4], dtype=torch.float32)

    graph[("src", "to", "dst")].edge_index = edge_index
    graph[("src", "to", "dst")].gauss_weight = edge_weight

    provider = ProjectionGraphProvider(
        graph=graph,
        edges_name=("src", "to", "dst"),
        edge_weight_attribute="gauss_weight",
        row_normalize=False,
        backend="torch",
    )

    edges = provider.get_edges()
    assert edges.layout == torch.sparse_csr

    matrix = edges.to_dense()
    assert matrix.shape == (graph["dst"].num_nodes, graph["src"].num_nodes)
    row_sums = matrix.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_projection_graph_provider_does_not_mutate_graph_weights() -> None:
    graph = _make_graph_with_edges()
    graph["data"].area = torch.tensor([2.0, 3.0, 4.0])
    original_weights = graph["data", "to", "target"].gauss_weight.clone()

    ProjectionGraphProvider(
        graph=graph,
        edges_name=("data", "to", "target"),
        edge_weight_attribute="gauss_weight",
        src_node_weight_attribute="area",
    )

    torch.testing.assert_close(graph["data", "to", "target"].gauss_weight, original_weights)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "file_path": "/unused/projection.npz",
            "edges_name": ("data", "to", "target"),
        },
    ],
)
def test_projection_graph_provider_requires_exactly_one_source(kwargs) -> None:
    with pytest.raises(ValueError, match="Exactly one of file_path or edges_name"):
        ProjectionGraphProvider(**kwargs)


def _make_graph_with_edges() -> HeteroData:
    graph = HeteroData()
    graph["data"].num_nodes = 3
    graph["target"].num_nodes = 2
    edge_index = torch.tensor([[0, 1, 2, 0], [0, 0, 1, 1]])
    edge_weight = torch.tensor([0.25, 0.75, 0.6, 0.4])
    graph[("data", "to", "target")].edge_index = edge_index
    graph[("data", "to", "target")].gauss_weight = edge_weight
    return graph


def test_from_config_returns_none_for_none() -> None:
    assert ProjectionGraphProvider.from_config(None) is None


def test_from_config_returns_none_for_empty_dict() -> None:
    assert ProjectionGraphProvider.from_config({}) is None


def test_from_config_file_mode(mocker) -> None:
    import numpy as np

    # rows do not sum to 1, so row_normalize (forwarded to the file path) is observable.
    mocker.patch(
        "anemoi.models.layers.graph_provider.load_npz",
        return_value=csr_matrix(np.array([[2.0, 2.0], [1.0, 3.0]])),
    )

    normalized = ProjectionGraphProvider.from_config(
        {"matrix_path": "/fake/path.npz", "row_normalize": True}, backend="torch"
    )
    assert isinstance(normalized, ProjectionGraphProvider)
    row_sums = normalized.get_edges().to_dense().sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    unnormalized = ProjectionGraphProvider.from_config(
        {"matrix_path": "/fake/path.npz", "row_normalize": False}, backend="torch"
    )
    row_sums = unnormalized.get_edges().to_dense().sum(dim=1)
    assert torch.allclose(row_sums, torch.tensor([4.0, 4.0]), atol=1e-6)


def test_from_config_edges_mode() -> None:
    graph = _make_graph_with_edges()
    provider = ProjectionGraphProvider.from_config(
        {
            "edges_name": ("data", "to", "target"),
            "edge_weight_attribute": "gauss_weight",
        },
        graph_data=graph,
        backend="torch",
    )
    assert isinstance(provider, ProjectionGraphProvider)
    matrix = provider.get_edges().to_dense()
    assert matrix.shape == (2, 3)


def test_from_config_edges_mode_requires_graph_data() -> None:
    with pytest.raises(ValueError, match="graph_data is required"):
        ProjectionGraphProvider.from_config({"edges_name": ("data", "to", "target")})


def test_from_config_ambiguous_raises() -> None:
    with pytest.raises(ValueError, match="at most one of"):
        ProjectionGraphProvider.from_config(
            {"matrix_path": "/fake/path.npz", "edges_name": ("data", "to", "target")},
        )


def test_from_config_invalid_raises() -> None:
    with pytest.raises(ValueError, match="must specify"):
        ProjectionGraphProvider.from_config({"unknown_key": "value"})


def test_projection_graph_provider_row_normalizes_csr_matrix() -> None:
    graph = HeteroData()
    graph["src"].num_nodes = 3
    graph["dst"].num_nodes = 2

    graph[("src", "to", "dst")].edge_index = torch.tensor([[0, 1, 2, 0], [0, 0, 1, 1]])
    graph[("src", "to", "dst")].gauss_weight = torch.tensor([2.0, 8.0, 6.0, 4.0])

    provider = ProjectionGraphProvider(
        graph=graph,
        edges_name=("src", "to", "dst"),
        edge_weight_attribute="gauss_weight",
        row_normalize=True,
        backend="torch",
    )

    edges = provider.get_edges()
    assert edges.layout == torch.sparse_csr

    expected = torch.tensor([[0.2, 0.8, 0.0], [0.4, 0.0, 0.6]])
    assert torch.allclose(edges.to_dense(), expected, atol=1e-6)


def test_projection_graph_provider_retains_requested_dtype() -> None:
    graph = HeteroData()
    graph["src"].num_nodes = 3
    graph["dst"].num_nodes = 2
    graph[("src", "to", "dst")].edge_index = torch.tensor([[0, 1, 2, 0], [0, 0, 1, 1]])

    provider = ProjectionGraphProvider(
        graph=graph,
        edges_name=("src", "to", "dst"),
        backend="torch",
    )

    float64_matrix = provider.get_edges(dtype=torch.float64)

    assert float64_matrix.dtype == torch.float64
    assert provider.get_edges(dtype=torch.float64) is float64_matrix


def test_projection_graph_provider_loads_npz_as_csr(tmp_path) -> None:
    file_path = tmp_path / "projection.npz"
    expected = torch.tensor([[0.25, 0.75, 0.0], [0.4, 0.0, 0.6]], dtype=torch.float32)
    save_npz(file_path, csr_matrix(expected.numpy()))

    provider = ProjectionGraphProvider(
        file_path=file_path,
        row_normalize=False,
        backend="torch",
    )

    edges = provider.get_edges()
    assert edges.layout == torch.sparse_csr
    assert torch.allclose(edges.to_dense(), expected)


def _uneven_provider(backend: str = "triton") -> ProjectionGraphProvider:
    """Four target nodes reading 2, 0, 3 and 1 of five source nodes."""
    graph = HeteroData()
    graph["src"].num_nodes = 5
    graph["dst"].num_nodes = 4
    graph[("src", "to", "dst")].edge_index = torch.tensor([[4, 1, 0, 2, 3, 2], [0, 0, 2, 2, 2, 3]])
    graph[("src", "to", "dst")].weight = torch.tensor([0.5, 0.25, 0.1, 0.2, 0.3, 1.0])
    return ProjectionGraphProvider(
        graph=graph, edges_name=("src", "to", "dst"), edge_weight_attribute="weight", backend=backend
    )


def _dense_from_table(neighbours: torch.Tensor, weights: torch.Tensor, num_sources: int) -> torch.Tensor:
    dense = torch.zeros(neighbours.shape[0], num_sources)
    dense.scatter_add_(1, neighbours.long(), weights)
    return dense


def test_projection_table_holds_the_matrix() -> None:
    neighbours, weights, row_lengths = _uneven_provider("triton").get_table()

    assert neighbours.dtype == row_lengths.dtype == torch.int32
    assert weights.dtype == torch.float32
    assert neighbours.shape == weights.shape == (4, 3)
    assert row_lengths.tolist() == [2, 0, 3, 1]
    expected = _uneven_provider("torch").get_edges().to_dense()
    torch.testing.assert_close(_dense_from_table(neighbours, weights, 5), expected)
    # The empty row has zero weights.
    assert (weights[1] == 0).all()


def test_each_backend_keeps_only_its_form_of_the_matrix() -> None:
    table_provider = _uneven_provider("triton")
    matrix_provider = _uneven_provider("torch")

    assert table_provider.shape == matrix_provider.shape == (4, 5)
    assert table_provider.projection_matrix is None
    assert matrix_provider.projection_table is None
    with pytest.raises(ValueError, match="get_table"):
        table_provider.get_edges()
    with pytest.raises(ValueError, match="get_edges"):
        matrix_provider.get_table()


def test_projection_table_built_in_bands_is_the_same(monkeypatch) -> None:
    from anemoi.models.layers import graph_provider

    expected = _uneven_provider().get_table()
    monkeypatch.setattr(graph_provider, "_TABLE_BAND_ENTRIES", 3)
    banded = _uneven_provider().get_table()

    for table, expected_table in zip(banded, expected):
        torch.testing.assert_close(table, expected_table)


def test_projection_table_refuses_tables_that_are_too_large(monkeypatch) -> None:
    from anemoi.models.layers import graph_provider

    # Four rows padded to the longest row of 3 make a table of 12 entries.
    monkeypatch.setattr(graph_provider, "_MAX_TABLE_ENTRIES", 12)
    _uneven_provider("triton")
    monkeypatch.setattr(graph_provider, "_MAX_TABLE_ENTRIES", 11)
    with pytest.raises(ValueError, match="too large"):
        _uneven_provider("triton")


def test_projection_table_is_shared_by_copies() -> None:
    import copy

    provider = _uneven_provider()
    table = provider.get_table()

    assert provider.get_table() is table
    copied = copy.deepcopy(provider)
    assert all(copied_table is shared for copied_table, shared in zip(copied.get_table(), table, strict=True))


def test_switching_to_the_torch_backend_gives_the_matrix_back() -> None:
    provider = _uneven_provider("triton")
    expected = _uneven_provider("torch")

    provider.use_torch_backend()

    assert provider.backend == "torch"
    assert provider.projection_table is None
    torch.testing.assert_close(provider.get_edges().to_dense(), expected.get_edges().to_dense())
    # Rows of 2, 0, 3 and 1 entries project as with the torch matrix.
    x = torch.randn(2, 5, 3)
    torch.testing.assert_close(SparseProjector()(x, provider), SparseProjector()(x, expected))


def test_use_torch_projections_switches_every_projection_in_a_model() -> None:
    model = torch.nn.ModuleDict(
        {
            "residual": _uneven_provider("triton"),
            "nested": torch.nn.Sequential(torch.nn.Linear(2, 2), _uneven_provider("triton")),
            "already_torch": _uneven_provider("torch"),
        }
    )

    use_torch_projections(model)

    providers = [module for module in model.modules() if isinstance(module, ProjectionGraphProvider)]
    assert len(providers) == 3
    assert all(provider.backend == "torch" and provider.projection_table is None for provider in providers)


def test_matrix_first_cast_in_inference_mode_can_be_used_in_training() -> None:
    # Lightning runs validation, the first use of the loss, under torch.inference_mode.
    provider = _uneven_provider("torch")
    x = torch.randn(2, 5, 3, dtype=torch.float64)
    with torch.inference_mode():
        SparseProjector()(x, provider, matrix_dtype=torch.float64)

    x = x.clone().requires_grad_(True)
    SparseProjector()(x, provider, matrix_dtype=torch.float64).sum().backward()

    assert x.grad is not None
