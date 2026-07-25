# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The pure-Python GraphBuilder object API (graph.md) + serialisation round-trip."""

import json

import numpy as np
import pytest
from torch_geometric.data import HeteroData

from anemoi.graphs.create import GraphBuilder
from anemoi.graphs.create import GraphCreator
from anemoi.graphs.edges import KNNEdges
from anemoi.graphs.edges.attributes import EdgeDirection
from anemoi.graphs.edges.attributes import EdgeLength
from anemoi.graphs.nodes import NPZFileNodes
from anemoi.utils.builder import build
from anemoi.utils.builder import to_dict


@pytest.fixture
def npz_file(tmp_path) -> str:
    path = str(tmp_path / "grid.npz")
    np.savez(path, latitudes=np.random.rand(40) * 180 - 90, longitudes=np.random.rand(40) * 360 - 180)
    return path


def _object_builder(npz: str) -> GraphBuilder:
    """A graph built with the graph.md constructor API (attributes as constructor kwargs)."""
    nodes = NPZFileNodes(npz_file=npz, name="data")
    edges = KNNEdges(
        source_name="data",
        target_name="data",
        num_nearest_neighbours=3,
        attributes=[EdgeLength(norm="unit-std"), EdgeDirection(norm="unit-std")],
    )
    return GraphBuilder(nodes=[nodes], edges=[edges])


def test_graph_builder_object_api_creates_graph(npz_file: str) -> None:
    graph = _object_builder(npz_file).create()
    assert isinstance(graph, HeteroData)
    assert "data" in graph.node_types
    edge = ("data", "to", "data")
    assert edge in graph.edge_types
    assert "edge_length" in graph[edge] and "edge_direction" in graph[edge]


def test_graph_builder_matches_config_path(npz_file: str) -> None:
    config = {
        "nodes": {"data": {"node_builder": {"_target_": "anemoi.graphs.nodes.NPZFileNodes", "npz_file": npz_file}}},
        "edges": [
            {
                "source_name": "data",
                "target_name": "data",
                "edge_builders": [{"_target_": "anemoi.graphs.edges.KNNEdges", "num_nearest_neighbours": 3}],
                "attributes": {
                    "edge_length": {"_target_": "anemoi.graphs.edges.attributes.EdgeLength", "norm": "unit-std"},
                    "edge_direction": {"_target_": "anemoi.graphs.edges.attributes.EdgeDirection", "norm": "unit-std"},
                },
            },
        ],
    }
    graph_cfg = GraphCreator(config=config).create()
    graph_obj = _object_builder(npz_file).create()

    assert graph_cfg.node_types == graph_obj.node_types
    assert graph_cfg.edge_types == graph_obj.edge_types
    edge = ("data", "to", "data")
    assert set(graph_cfg[edge].edge_attrs()) == set(graph_obj[edge].edge_attrs())


def test_graph_builder_serialisation_round_trip(npz_file: str) -> None:
    builder = _object_builder(npz_file)
    spec = to_dict(builder)
    rebuilt = build(json.loads(json.dumps(spec)))  # JSON-able -> rebuilt object

    assert isinstance(rebuilt, GraphBuilder)
    assert rebuilt.nodes[0].name == "data"
    assert rebuilt.edges[0].num_nearest_neighbours == 3
    assert list(rebuilt.edges[0].attributes) == list(builder.edges[0].attributes)
    # and the rebuilt object still creates the same graph
    graph = rebuilt.create()
    assert ("data", "to", "data") in graph.edge_types
