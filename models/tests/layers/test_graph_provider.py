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


def _make_static_graph_provider() -> StaticGraphProvider:
    graph = HeteroData()
    graph.edge_index = torch.tensor([[0, 1, 2, 0], [1, 0, 1, 0]], dtype=torch.long)
    graph.edge_attr = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32)

    return StaticGraphProvider(
        graph=graph,
        edge_attributes=["edge_attr"],
        src_size=3,
        dst_size=2,
        trainable_size=2,
    )


def test_static_graph_provider_permutes_legacy_trainable_state_on_load() -> None:
    provider = _make_static_graph_provider()
    legacy_trainable = torch.tensor(
        [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]],
        dtype=torch.float32,
    )
    legacy_state_dict = {"trainable.trainable": legacy_trainable.clone()}

    provider.load_state_dict(legacy_state_dict, strict=True)

    expected = legacy_trainable.index_select(0, provider.perm)
    assert torch.equal(provider.trainable.trainable, expected)


def test_static_graph_provider_does_not_repermute_current_trainable_state() -> None:
    provider = _make_static_graph_provider()
    current_trainable = torch.tensor(
        [[100.0, 101.0], [200.0, 201.0], [300.0, 301.0], [400.0, 401.0]],
        dtype=torch.float32,
    )
    provider.trainable.trainable.data.copy_(current_trainable)

    state_dict = provider.state_dict()

    reloaded_provider = _make_static_graph_provider()
    reloaded_provider.load_state_dict(state_dict, strict=True)

    assert torch.equal(reloaded_provider.trainable.trainable, current_trainable)
