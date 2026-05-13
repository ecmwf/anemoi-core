from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.models.layers.residual import ScalarOrnsteinConnection
from anemoi.models.layers.residual import SkipConnection
from anemoi.models.layers.residual import SpectralOrnsteinConnection
from anemoi.models.layers.residual import TruncatedConnection

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def graph_data():
    g = HeteroData()
    g["data"].num_nodes = 2
    g["hidden"].num_nodes = 1
    g["hidden", "to", "data"].edge_index = torch.tensor([[0, 0], [0, 1]])
    g["hidden", "to", "data"].edge_length = torch.tensor([1.0, 2.0])
    g["hidden", "to", "data"].gauss_weight = torch.tensor([0.5, 0.5])
    g["data", "to", "hidden"].edge_index = torch.tensor([[0, 1], [0, 0]])
    g["data", "to", "hidden"].edge_length = torch.tensor([1.0, 2.0])
    g["data", "to", "hidden"].gauss_weight = torch.tensor([0.5, 0.5])
    g["data"].weight = torch.tensor([1.0, 0.5])  # Example weights for data nodes
    g["hidden"].weight = torch.tensor([0.8])  # Example weight for hidden node
    return g


@pytest.fixture
def flat_data():
    x = torch.randn(11, 7, 5, 2, 3)  # batch, dates, ensemble, grid, features
    return x


@pytest.fixture
def edge_index():
    return torch.tensor([[0, 1, 1], [1, 0, 2]])


def _make_data_indices(n_prognostic=3):
    """Create a mock data_indices with n_prognostic prognostic variables."""
    data_indices = MagicMock()
    data_indices.model.input.prognostic = list(range(n_prognostic))
    names = {f"var{i}": i for i in range(n_prognostic)}
    data_indices.model.input.name_to_index = names
    data_indices.data.input.prognostic = list(range(n_prognostic))
    return data_indices


def _make_regular_graph(nlat=8):
    """Create a HeteroData graph for a regular lat-lon grid."""
    nlon = 2 * nlat
    lats = np.linspace(-90, 90, nlat)
    lons = np.linspace(0, 360, nlon, endpoint=False)
    grid = np.array([(la, lo) for la in lats for lo in lons])
    graph = HeteroData()
    graph["data"].x = torch.from_numpy(grid).float()
    return graph, nlat, nlon


def _make_octahedral_graph(nlat=8):
    """Create a HeteroData graph for an octahedral grid."""
    oct_lons = [20 + 4 * i for i in range(nlat // 2)]
    oct_lons = oct_lons + oct_lons[::-1]
    n_points = sum(oct_lons)
    coords = torch.zeros(n_points, 2)
    idx = 0
    for i, nl in enumerate(oct_lons):
        coords[idx : idx + nl, 0] = float(i)
        coords[idx : idx + nl, 1] = torch.arange(nl).float()
        idx += nl
    graph = HeteroData()
    graph["data"].x = coords
    return graph, nlat, n_points


# ── TruncatedConnection tests (existing) ─────────────────────────────────


def test_truncation_mapper_init(graph_data):
    _ = TruncatedConnection(
        graph_data,
        truncation_down_edges_name=("data", "to", "hidden"),
        truncation_up_edges_name=("hidden", "to", "data"),
        edge_weight_attribute="edge_length",
    )


def test_forward(graph_data):
    mapper = TruncatedConnection(
        graph_data,
        truncation_down_edges_name=("data", "to", "hidden"),
        truncation_up_edges_name=("hidden", "to", "data"),
        edge_weight_attribute="edge_length",
    )
    x = torch.randn(5, 2, 2, 2, 3)  # (batch, dates, ensemble, grid, features)
    x_truncated = mapper.forward(x)
    assert x_truncated.shape == (5, 2, 2, 3)  # (batch, ensemble, coarse_grid, features)


def test_forward_no_weight(graph_data):
    mapper = TruncatedConnection(
        graph_data,
        truncation_down_edges_name=("data", "to", "hidden"),
        truncation_up_edges_name=("hidden", "to", "data"),
    )
    x = torch.randn(5, 2, 2, 2, 3)  # (batch, dates, ensemble, grid, features)
    x_truncated = mapper.forward(x)
    assert x_truncated.shape == (5, 2, 2, 3)  # (batch, ensemble, coarse_grid, features)


def test_forward_with_src_node_weight(graph_data):
    mapper = TruncatedConnection(
        graph_data,
        truncation_down_edges_name=("data", "to", "hidden"),
        truncation_up_edges_name=("hidden", "to", "data"),
        src_node_weight_attribute="weight",
    )
    x = torch.randn(5, 2, 2, 2, 3)  # (batch, dates, ensemble, grid, features)
    x_truncated = mapper.forward(x)
    assert x_truncated.shape == (5, 2, 2, 3)  # (batch, ensemble, coarse_grid, features)


def test_forward_with_edges_name(graph_data):
    mapper = TruncatedConnection(
        graph_data,
        truncation_down_edges_name=("data", "to", "hidden"),
        truncation_up_edges_name=("hidden", "to", "data"),
        edge_weight_attribute="edge_length",
    )
    x = torch.randn(5, 2, 2, 2, 3)  # (batch, dates, ensemble, grid, features)
    x_truncated = mapper.forward(x)
    assert x_truncated.shape == (5, 2, 2, 3)  # (batch, ensemble, coarse_grid, features)


def test_truncated_connection_shard_sizes_calls_all_to_all(graph_data, monkeypatch):
    mapper = TruncatedConnection(
        graph_data,
        truncation_down_edges_name=("data", "to", "hidden"),
        truncation_up_edges_name=("hidden", "to", "data"),
        edge_weight_attribute="edge_length",
    )
    calls = []

    def fake_all_to_all(x, scatter_dim, scatter_sizes, gather_dim, gather_sizes, group):
        calls.append(
            {
                "scatter_dim": scatter_dim,
                "scatter_sizes": scatter_sizes,
                "gather_dim": gather_dim,
                "gather_sizes": gather_sizes,
            }
        )
        return x

    monkeypatch.setattr("anemoi.models.layers.residual.all_to_all_transpose", fake_all_to_all)
    monkeypatch.setattr("anemoi.models.layers.residual.get_shard_sizes", lambda x, dim, group: [x.shape[dim]])

    x = torch.randn(3, 2, 1, 2, 3)  # (batch, dates, ensemble, grid, features)
    mapper.forward(x, grid_shard_sizes=[1, 1])

    # Two all-to-all calls: grid->channel before projection, channel->grid after
    assert len(calls) == 2
    assert calls[0]["gather_sizes"] == [1, 1]  # grid_shard_sizes used as gather
    assert calls[1]["scatter_sizes"] == [1, 1]  # grid_shard_sizes used as scatter


def test_skipconnection(flat_data):
    mapper = SkipConnection()
    out = mapper.forward(flat_data)
    expected_out = flat_data[:, -1, ...]

    assert torch.allclose(out, expected_out), "SkipConnection did not return the expected output."


# ── ScalarOrnsteinConnection tests ───────────────────────────────────────


def test_scalar_ornstein_shape():
    data_indices = _make_data_indices(3)
    conn = ScalarOrnsteinConnection(data_indices=data_indices)
    x = torch.randn(2, 4, 1, 10, 3)  # batch, time, ens, nodes, features
    out = conn.forward(x)
    assert out.shape == (2, 1, 10, 3)


def test_scalar_ornstein_frozen_weights():
    data_indices = _make_data_indices(3)
    conn = ScalarOrnsteinConnection(theta_train=False, data_indices=data_indices)
    assert not conn.weight.requires_grad


def test_scalar_ornstein_trainable_weights():
    data_indices = _make_data_indices(3)
    conn = ScalarOrnsteinConnection(theta_train=True, data_indices=data_indices)
    assert conn.weight.requires_grad


# ── SpectralOrnsteinConnection tests ─────────────────────────────────────


@pytest.mark.parametrize("truncate", [False, True])
def test_spectral_ornstein_regular(truncate):
    graph, nlat, nlon = _make_regular_graph(nlat=8)
    data_indices = _make_data_indices(3)
    conn = SpectralOrnsteinConnection(
        lmax=2,
        grid="regular",
        truncate=truncate,
        graph=graph,
        data_indices=data_indices,
        dataset_name="data",
    )
    x = torch.randn(2, 3, 1, nlat * nlon, 3)
    out = conn.forward(x)
    assert out.shape == (2, 1, nlat * nlon, 3)


@pytest.mark.parametrize("truncate", [False, True])
def test_spectral_ornstein_octahedral(truncate):
    graph, nlat, n_points = _make_octahedral_graph(nlat=8)
    data_indices = _make_data_indices(3)
    conn = SpectralOrnsteinConnection(
        lmax=2,
        grid="octahedral",
        truncate=truncate,
        graph=graph,
        data_indices=data_indices,
        dataset_name="data",
    )
    x = torch.randn(2, 3, 1, n_points, 3)
    out = conn.forward(x)
    assert out.shape == (2, 1, n_points, 3)


def test_spectral_ornstein_skip_truncate_variables():
    graph, nlat, nlon = _make_regular_graph(nlat=8)
    data_indices = _make_data_indices(3)
    conn = SpectralOrnsteinConnection(
        lmax=2,
        grid="regular",
        truncate=True,
        skip_truncate_variables=["var0"],
        graph=graph,
        data_indices=data_indices,
        dataset_name="data",
    )
    # var0 (index 0) should be excluded from truncation
    assert 0 not in conn._truncation_input_idx
    assert 1 in conn._truncation_input_idx
    assert 2 in conn._truncation_input_idx

    x = torch.randn(2, 3, 1, nlat * nlon, 3)
    out = conn.forward(x)
    assert out.shape == (2, 1, nlat * nlon, 3)


def test_spectral_ornstein_no_truncation_has_no_filter():
    graph, nlat, nlon = _make_regular_graph(nlat=8)
    data_indices = _make_data_indices(3)
    conn = SpectralOrnsteinConnection(
        lmax=2,
        grid="regular",
        truncate=False,
        graph=graph,
        data_indices=data_indices,
        dataset_name="data",
    )
    assert not hasattr(conn, "filter")
    assert not hasattr(conn, "x_fsht")


def test_spectral_ornstein_truncation_has_filter():
    graph, nlat, nlon = _make_regular_graph(nlat=8)
    data_indices = _make_data_indices(3)
    conn = SpectralOrnsteinConnection(
        lmax=2,
        grid="regular",
        truncate=True,
        graph=graph,
        data_indices=data_indices,
        dataset_name="data",
    )
    assert hasattr(conn, "filter")
    assert hasattr(conn, "x_fsht")
    assert hasattr(conn, "x_isht")
