# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.edges import KNNEdges
from anemoi.graphs.edges import MutualKNNEdges
from anemoi.graphs.edges import ReversedKNNEdges


def _edge_set(edge_index: torch.Tensor) -> set:
    """Return the set of (source, target) tuples in an edge_index tensor."""
    return {tuple(col) for col in edge_index.t().tolist()}


def test_init():
    """Test MutualKNNEdges initialization."""
    MutualKNNEdges("test_nodes1", "test_nodes2", 3)


def test_init_with_reversed_k():
    """An explicit reversed k is stored as given."""
    builder = MutualKNNEdges("test_nodes1", "test_nodes2", 3, reversed_num_nearest_neighbours=5)
    assert builder.num_nearest_neighbours == 3
    assert builder.reversed_num_nearest_neighbours == 5


def test_default_reversed_k_matches_forward():
    """Reversed k defaults to the forward k when not provided."""
    builder = MutualKNNEdges("test_nodes1", "test_nodes2", 4)
    assert builder.reversed_num_nearest_neighbours == 4


@pytest.mark.parametrize("num_nearest_neighbours", [-1, 0, 2.6, "hello", None])
def test_fail_init(num_nearest_neighbours):
    """Invalid num_nearest_neighbours raises AssertionError."""
    with pytest.raises(AssertionError):
        MutualKNNEdges("test_nodes1", "test_nodes2", num_nearest_neighbours)


@pytest.mark.parametrize("reversed_num_nearest_neighbours", [-1, 0, 2.6, "hello"])
def test_fail_init_reversed(reversed_num_nearest_neighbours):
    """Invalid reversed_num_nearest_neighbours raises AssertionError."""
    with pytest.raises(AssertionError):
        MutualKNNEdges("test_nodes1", "test_nodes2", 3, reversed_num_nearest_neighbours=reversed_num_nearest_neighbours)


def test_mutual_knn(graph_with_nodes):
    """MutualKNNEdges registers the expected edge type."""
    builder = MutualKNNEdges("test_nodes", "test_nodes", 3)
    graph = builder.update_graph(graph_with_nodes)
    assert ("test_nodes", "to", "test_nodes") in graph.edge_types


def test_mutual_equals_intersection(graph_with_nodes):
    """Mutual edges are exactly KNN edges intersected with ReversedKNN edges."""
    k = 3
    nodes = graph_with_nodes["test_nodes"]
    mutual = MutualKNNEdges("test_nodes", "test_nodes", k).compute_edge_index(nodes, nodes)
    forward = KNNEdges("test_nodes", "test_nodes", k).compute_edge_index(nodes, nodes)
    reversed_ = ReversedKNNEdges("test_nodes", "test_nodes", k).compute_edge_index(nodes, nodes)

    assert _edge_set(mutual) == _edge_set(forward) & _edge_set(reversed_)


def test_mutual_is_subset_of_knn(graph_with_nodes):
    """Every mutual edge is also a plain KNN edge."""
    k = 4
    nodes = graph_with_nodes["test_nodes"]
    mutual = MutualKNNEdges("test_nodes", "test_nodes", k).compute_edge_index(nodes, nodes)
    forward = KNNEdges("test_nodes", "test_nodes", k).compute_edge_index(nodes, nodes)

    assert _edge_set(mutual).issubset(_edge_set(forward))


def test_mutual_knn_masking(graph_with_nodes):
    """Masked edges stay within the masked node set and indices are remapped."""
    builder = MutualKNNEdges(
        "test_nodes",
        "test_nodes",
        3,
        source_mask_attr_name="mask2",
        target_mask_attr_name="mask2",
    )
    nodes = graph_with_nodes["test_nodes"]
    edge_index = builder.compute_edge_index(nodes, nodes)

    mask2 = nodes["mask2"].squeeze()
    assert mask2[edge_index[0]].all()
    assert mask2[edge_index[1]].all()


def test_mutual_knn_sklearn_fallback(monkeypatch, graph_with_nodes):
    """The scikit-learn fallback yields the same mutual edges as the torch-cluster path."""
    import anemoi.graphs.edges.builders.base as base_module

    nodes = graph_with_nodes["test_nodes"]
    builder = MutualKNNEdges("test_nodes", "test_nodes", 3)
    primary_edges = builder.compute_edge_index(nodes, nodes)

    monkeypatch.setattr(base_module, "TORCH_CLUSTER_AVAILABLE", False)
    fallback_edges = builder.compute_edge_index(nodes, nodes)

    assert _edge_set(primary_edges) == _edge_set(fallback_edges)


# ---------------------------------------------------------------------------
# Heterogeneous (distinct source vs. target) node-set fixtures and tests
# ---------------------------------------------------------------------------

# src_nodes: 3x4 grid (same layout as the global fixture), scaled to radians.
_SRC_LATS = np.array([-0.15, 0.0, 0.15])
_SRC_LONS = np.array([0.0, 0.25, 0.5, 0.75])
_SRC_COORDS = np.array([[lat, lon] for lat in _SRC_LATS for lon in _SRC_LONS])  # 12 nodes

# tgt_nodes: 2x4 coarser grid — different shape and values from src_nodes.
_TGT_LATS = np.array([-0.3, 0.3])
_TGT_LONS = np.array([0.125, 0.375, 0.625, 0.875])
_TGT_COORDS = np.array([[lat, lon] for lat in _TGT_LATS for lon in _TGT_LONS])  # 8 nodes


@pytest.fixture
def graph_with_two_node_sets() -> HeteroData:
    """Graph with two distinct node sets: 12 src_nodes and 8 tgt_nodes."""
    graph = HeteroData()
    graph["src_nodes"].x = 2 * torch.pi * torch.tensor(_SRC_COORDS, dtype=torch.float64)
    graph["tgt_nodes"].x = 2 * torch.pi * torch.tensor(_TGT_COORDS, dtype=torch.float64)
    return graph


def test_mutual_equals_intersection_heterogeneous(graph_with_two_node_sets):
    """Intersection invariant holds for distinct source/target sets and asymmetric k."""
    k_fwd, k_rev = 3, 2
    src = graph_with_two_node_sets["src_nodes"]
    tgt = graph_with_two_node_sets["tgt_nodes"]

    mutual = MutualKNNEdges("src_nodes", "tgt_nodes", k_fwd, reversed_num_nearest_neighbours=k_rev).compute_edge_index(
        src, tgt
    )
    forward = KNNEdges("src_nodes", "tgt_nodes", k_fwd).compute_edge_index(src, tgt)
    reversed_ = ReversedKNNEdges("src_nodes", "tgt_nodes", k_rev).compute_edge_index(src, tgt)

    assert _edge_set(mutual) == _edge_set(forward) & _edge_set(reversed_)


def test_mutual_knn_sklearn_fallback_heterogeneous(monkeypatch, graph_with_two_node_sets):
    """The scikit-learn fallback agrees with the torch-cluster path for distinct node sets and asymmetric k."""
    import anemoi.graphs.edges.builders.base as base_module

    k_fwd, k_rev = 3, 2
    src = graph_with_two_node_sets["src_nodes"]
    tgt = graph_with_two_node_sets["tgt_nodes"]
    builder = MutualKNNEdges("src_nodes", "tgt_nodes", k_fwd, reversed_num_nearest_neighbours=k_rev)

    primary = builder.compute_edge_index(src, tgt)

    monkeypatch.setattr(base_module, "TORCH_CLUSTER_AVAILABLE", False)
    fallback = builder.compute_edge_index(src, tgt)

    assert _edge_set(primary) == _edge_set(fallback)


def test_schema_validates_mutual_knn():
    """A MutualKNNEdges block validates against the edge builder schema union."""
    from pydantic import TypeAdapter

    from anemoi.graphs.schemas.edge_schemas import EdgeBuilderSchemas

    adapter = TypeAdapter(EdgeBuilderSchemas)
    model = adapter.validate_python(
        {
            "_target_": "anemoi.graphs.edges.MutualKNNEdges",
            "num_nearest_neighbours": 3,
            "reversed_num_nearest_neighbours": 5,
        }
    )
    assert model.num_nearest_neighbours == 3
    assert model.reversed_num_nearest_neighbours == 5


def test_schema_rejects_invalid_mutual_knn():
    """An invalid MutualKNNEdges block fails schema validation."""
    from pydantic import TypeAdapter
    from pydantic import ValidationError

    from anemoi.graphs.schemas.edge_schemas import EdgeBuilderSchemas

    adapter = TypeAdapter(EdgeBuilderSchemas)
    with pytest.raises(ValidationError):
        adapter.validate_python({"_target_": "anemoi.graphs.edges.MutualKNNEdges", "num_nearest_neighbours": -1})


def test_mutual_knn_graph_creation(tmp_path, mock_grids_path):
    """An end-to-end recipe using MutualKNNEdges builds a graph via GraphCreator."""
    import yaml

    from anemoi.graphs.create import GraphCreator

    grids_path, _ = mock_grids_path
    cfg = {
        "nodes": {
            "test_nodes": {
                "node_builder": {
                    "_target_": "anemoi.graphs.nodes.NPZFileNodes",
                    "npz_file": grids_path + "/grid-o16.npz",
                },
            },
        },
        "edges": [
            {
                "source_name": "test_nodes",
                "target_name": "test_nodes",
                "edge_builders": [
                    {
                        "_target_": "anemoi.graphs.edges.MutualKNNEdges",
                        "num_nearest_neighbours": 3,
                    },
                ],
            },
        ],
    }
    # The same edge-builder block fed to GraphCreator also passes recipe-schema validation.
    from pydantic import TypeAdapter

    from anemoi.graphs.schemas.edge_schemas import EdgeBuilderSchemas

    TypeAdapter(EdgeBuilderSchemas).validate_python(cfg["edges"][0]["edge_builders"][0])

    config_path = tmp_path / "mutual_knn_config.yaml"
    with config_path.open("w") as file:
        yaml.dump(cfg, file)

    graph = GraphCreator(config=config_path).create()
    assert ("test_nodes", "to", "test_nodes") in graph.edge_types
