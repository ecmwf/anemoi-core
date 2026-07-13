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
from torch_geometric.data import HeteroData

from anemoi.models.layers.graph_provider import ProjectionGraphProvider


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

    matrix = provider.get_edges().to_dense()
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

    matrix = provider.get_edges().to_dense()
    assert matrix.shape == (graph["dst"].num_nodes, graph["src"].num_nodes)
    row_sums = matrix.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


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
    from scipy.sparse import csr_matrix

    # rows do not sum to 1, so row_normalize (forwarded to the file path) is observable.
    mocker.patch("scipy.sparse.load_npz", return_value=csr_matrix(np.array([[2.0, 2.0], [1.0, 3.0]])))

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
        {"edges_name": ("data", "to", "target"), "edge_weight_attribute": "gauss_weight"},
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
