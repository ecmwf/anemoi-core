# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import io
from collections import OrderedDict

import pytest
import torch
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from torch_geometric.data import HeteroData

from anemoi.models.layers.graph_provider import DynamicGraphProvider
from anemoi.models.layers.graph_provider import NoOpGraphProvider
from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.models.layers.graph_provider import StaticGraphProvider


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

    normalized = ProjectionGraphProvider.from_config({"matrix_path": "/fake/path.npz", "row_normalize": True})
    assert isinstance(normalized, ProjectionGraphProvider)
    row_sums = normalized.get_edges().to_dense().sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    unnormalized = ProjectionGraphProvider.from_config({"matrix_path": "/fake/path.npz", "row_normalize": False})
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
    )

    edges = provider.get_edges()
    assert edges.layout == torch.sparse_csr
    assert torch.allclose(edges.to_dense(), expected)


def _static_graph_provider(topology_cache_size: int = 0):
    graph = HeteroData()
    graph["src", "to", "dst"].edge_index = torch.tensor([[0, 1], [0, 1]])
    graph["src", "to", "dst"].edge_attr = torch.ones(2, 1)
    return StaticGraphProvider(
        graph=graph["src", "to", "dst"],
        edge_attributes=["edge_attr"],
        src_size=2,
        dst_size=2,
        trainable_size=0,
        topology_cache_size=topology_cache_size,
    )


@pytest.mark.parametrize("provider", [DynamicGraphProvider(edge_dim=1), NoOpGraphProvider()])
def test_non_static_graph_provider_does_not_keep_topology(provider) -> None:
    calls = []

    def build_fn():
        calls.append(1)
        return object()

    first = provider.cached_topology("key", build_fn)
    second = provider.cached_topology("key", build_fn)

    assert first is not second
    assert len(calls) == 2


def test_static_graph_provider_keeps_topology_for_same_key() -> None:
    provider = _static_graph_provider()
    calls = []

    def build_fn():
        calls.append(1)
        return object()

    first = provider.cached_topology("a", build_fn)
    assert provider.cached_topology("a", build_fn) is first
    assert len(calls) == 1

    assert provider.cached_topology("b", build_fn) is not first
    assert len(calls) == 2


def test_static_graph_provider_does_not_pickle_topology() -> None:
    provider = _static_graph_provider()
    provider.cached_topology("a", lambda: torch.arange(3))

    buffer = io.BytesIO()
    torch.save(provider, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)

    assert provider._topology_data is not None
    assert loaded._topology_key is None
    assert loaded._topology_data is None


def test_static_graph_provider_loads_pickle_without_topology_attributes() -> None:
    provider = _static_graph_provider()
    state = provider.__getstate__()
    del state["_topology_key"]
    del state["_topology_data"]

    loaded = provider.__class__.__new__(provider.__class__)
    loaded.__setstate__(state)

    assert loaded.cached_topology("a", lambda: 1) == 1


@pytest.fixture(autouse=True)
def reset_topology_cache_state(monkeypatch):
    """Start each test with no recorded graphs and no warning."""
    monkeypatch.setattr(StaticGraphProvider, "_graphs_with_topology", OrderedDict())
    monkeypatch.setattr(StaticGraphProvider, "_warned_no_topology_cache_size", False)


def test_topology_cache_size_removes_least_recently_used() -> None:
    providers = [_static_graph_provider(topology_cache_size=2) for _ in range(3)]

    providers[0].cached_topology("a", lambda: 0)
    providers[1].cached_topology("a", lambda: 1)
    providers[0].cached_topology("a", lambda: 0)  # providers[1] is now the least recently used
    providers[2].cached_topology("a", lambda: 2)

    assert providers[0]._topology_data == 0
    assert providers[1]._topology_data is None
    assert providers[2]._topology_data == 2


def test_topology_cache_size_warns_once_when_disabled(caplog) -> None:
    providers = [_static_graph_provider() for _ in range(3)]

    with caplog.at_level("WARNING", logger="anemoi.models.layers.graph_provider"):
        for i, provider in enumerate(providers):
            provider.cached_topology("a", lambda i=i: i)

    assert [provider._topology_data for provider in providers] == [0, 1, 2]
    assert sum("topology_cache_size" in record.message for record in caplog.records) == 1


def test_topology_cache_size_does_not_warn_when_enabled(caplog) -> None:
    provider = _static_graph_provider(topology_cache_size=4)

    with caplog.at_level("WARNING", logger="anemoi.models.layers.graph_provider"):
        provider.cached_topology("a", lambda: 0)

    assert not any("topology_cache_size" in record.message for record in caplog.records)
