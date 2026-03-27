# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

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

    assert provider.get_edges().layout == torch.sparse_csr
    matrix = provider.get_edges().to_dense()
    assert matrix.shape == (graph["dst"].num_nodes, graph["src"].num_nodes)

    row_sums = matrix.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_projection_graph_provider_fewer_dst_than_src() -> None:
    graph = HeteroData()
    graph["src"].num_nodes = 4
    graph["dst"].num_nodes = 2

    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_weight = torch.tensor([0.3, 0.7, 0.4, 0.6])  # per-target sums: [1.0, 1.0]

    graph[("src", "to", "dst")].edge_index = edge_index
    graph[("src", "to", "dst")].gauss_weight = edge_weight

    provider = ProjectionGraphProvider(
        graph=graph,
        edges_name=("src", "to", "dst"),
        edge_weight_attribute="gauss_weight",
        row_normalize=False,
    )

    assert provider.get_edges().layout == torch.sparse_csr
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

    assert provider.get_edges().layout == torch.sparse_csr
    matrix = provider.get_edges().to_dense()
    assert matrix.shape == (graph["dst"].num_nodes, graph["src"].num_nodes)
    row_sums = matrix.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_projection_graph_provider_can_override_sparse_format_to_coo() -> None:
    graph = HeteroData()
    graph["src"].num_nodes = 3
    graph["dst"].num_nodes = 2
    graph[("src", "to", "dst")].edge_index = torch.tensor([[0, 1, 2, 0], [0, 0, 1, 1]], dtype=torch.int32)
    graph[("src", "to", "dst")].gauss_weight = torch.tensor([0.25, 0.75, 0.6, 0.4], dtype=torch.float32)

    provider = ProjectionGraphProvider(
        graph=graph,
        edges_name=("src", "to", "dst"),
        edge_weight_attribute="gauss_weight",
        row_normalize=False,
        sparse_format="coo",
    )

    matrix = provider.get_edges()
    assert matrix.layout == torch.sparse_coo
    assert torch.allclose(matrix.to_dense().sum(dim=1), torch.ones(graph["dst"].num_nodes), atol=1e-6)


def test_projection_graph_provider_csr_and_coo_are_equivalent() -> None:
    graph = HeteroData()
    graph["src"].num_nodes = 4
    graph["dst"].num_nodes = 3

    graph[("src", "to", "dst")].edge_index = torch.tensor(
        [[0, 1, 2, 3, 0, 2], [0, 0, 1, 1, 2, 2]],
        dtype=torch.int32,
    )
    graph[("src", "to", "dst")].gauss_weight = torch.tensor([0.2, 0.8, 0.25, 0.75, 0.6, 0.4], dtype=torch.float32)

    provider_csr = ProjectionGraphProvider(
        graph=graph,
        edges_name=("src", "to", "dst"),
        edge_weight_attribute="gauss_weight",
        row_normalize=False,
        sparse_format="csr",
    )
    provider_coo = ProjectionGraphProvider(
        graph=graph,
        edges_name=("src", "to", "dst"),
        edge_weight_attribute="gauss_weight",
        row_normalize=False,
        sparse_format="coo",
    )

    assert provider_csr.get_edges().layout == torch.sparse_csr
    assert provider_coo.get_edges().layout == torch.sparse_coo
    assert torch.equal(provider_csr.get_edges().to_dense(), provider_coo.get_edges().to_dense())


def test_projection_graph_provider_row_normalization_matches_between_csr_and_coo() -> None:
    graph = HeteroData()
    graph["src"].num_nodes = 4
    graph["dst"].num_nodes = 3

    graph[("src", "to", "dst")].edge_index = torch.tensor(
        [[0, 1, 2, 3, 0, 2], [0, 0, 1, 1, 2, 2]],
        dtype=torch.int32,
    )
    graph[("src", "to", "dst")].gauss_weight = torch.tensor([2.0, 8.0, 1.0, 3.0, 6.0, 4.0], dtype=torch.float32)

    provider_csr = ProjectionGraphProvider(
        graph=graph,
        edges_name=("src", "to", "dst"),
        edge_weight_attribute="gauss_weight",
        row_normalize=True,
        sparse_format="csr",
    )
    provider_coo = ProjectionGraphProvider(
        graph=graph,
        edges_name=("src", "to", "dst"),
        edge_weight_attribute="gauss_weight",
        row_normalize=True,
        sparse_format="coo",
    )

    matrix_csr = provider_csr.get_edges().to_dense()
    matrix_coo = provider_coo.get_edges().to_dense()
    assert torch.equal(matrix_csr, matrix_coo)
    assert torch.equal(matrix_csr.sum(dim=1), torch.ones(graph["dst"].num_nodes))
