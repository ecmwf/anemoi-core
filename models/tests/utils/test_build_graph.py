# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path

import pytest
import torch

from anemoi.models.utils.graphs import build_graph
from anemoi.utils.config import DotDict

LATITUDES = [0.0, 10.0, -10.0, 45.0, -45.0, 80.0]
LONGITUDES = [0.0, 90.0, 180.0, -90.0, 30.0, -150.0]
KNN = {"_target_": "anemoi.graphs.edges.KNNEdges", "num_nearest_neighbours": 2}


def _nodes() -> dict:
    return {
        "grid": {
            "node_builder": {
                "_target_": "anemoi.graphs.nodes.LatLonNodes",
                "latitudes": LATITUDES,
                "longitudes": LONGITUDES,
            }
        },
        "obs": {"node_builder": {"_target_": "anemoi.graphs.nodes.LatLonNodes"}, "attributes": {}},
        "hidden": {"node_builder": {"_target_": "anemoi.graphs.nodes.TriNodes", "resolution": 1}},
    }


def _edges(source: str, target: str) -> dict:
    return {"source_name": source, "target_name": target, "edge_builders": [KNN], "attributes": {}}


def _graph_config(edges: list[tuple[str, str]], nodes: list[str] | None = None) -> DotDict:
    all_nodes = _nodes()
    return DotDict(
        {
            "nodes": {name: all_nodes[name] for name in (nodes if nodes is not None else all_nodes)},
            "edges": [_edges(source, target) for source, target in edges],
        }
    )


IS_STATIC = {"grid": True, "obs": False}


def test_build_graph_splits_static_and_dynamic_and_saves(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.pt"
    graph, dynamic = build_graph(
        _graph_config([("grid", "hidden"), ("obs", "hidden"), ("hidden", "grid")]),
        IS_STATIC,
        "hidden",
        graph_path=graph_path,
    )

    assert set(graph.node_types) == {"grid", "hidden"}
    assert set(graph.edge_types) == {("grid", "to", "hidden"), ("hidden", "to", "grid")}
    assert list(dynamic.nodes) == ["obs"]
    assert dynamic.edges[("obs", "to", "hidden")]["edge_builders"]
    assert dynamic.edges[("grid", "to", "hidden")] == {}
    assert graph_path.exists()


def test_build_graph_fills_static_parts_missing_from_loaded_graph(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.pt"
    build_graph(_graph_config([("grid", "hidden")]), IS_STATIC, "hidden", graph_path=graph_path)
    stored = torch.load(graph_path, weights_only=False)

    graph, _ = build_graph(
        _graph_config([("grid", "hidden"), ("hidden", "grid")]), IS_STATIC, "hidden", graph_path=graph_path
    )

    # The loaded edges are kept, the missing ones are built, and the file is left untouched.
    assert torch.equal(graph["grid", "to", "hidden"].edge_index, stored["grid", "to", "hidden"].edge_index)
    assert ("hidden", "to", "grid") in graph.edge_types
    assert ("hidden", "to", "grid") not in torch.load(graph_path, weights_only=False).edge_types


def test_build_graph_overwrite_rebuilds_the_stored_graph(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.pt"
    build_graph(_graph_config([("grid", "hidden")]), IS_STATIC, "hidden", graph_path=graph_path)

    config = DotDict({**_graph_config([("hidden", "grid")]), "overwrite": True})
    graph, _ = build_graph(config, IS_STATIC, "hidden", graph_path=graph_path)

    assert set(graph.edge_types) == {("hidden", "to", "grid")}
    assert set(torch.load(graph_path, weights_only=False).edge_types) == {("hidden", "to", "grid")}


def test_build_graph_loads_existing_graph_without_nodes_in_config(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.pt"
    build_graph(_graph_config([("grid", "hidden")]), IS_STATIC, "hidden", graph_path=graph_path)

    graph, dynamic = build_graph(DotDict({"overwrite": False}), {"grid": True}, "hidden", graph_path=graph_path)

    assert set(graph.node_types) == {"grid", "hidden"}
    assert set(graph.edge_types) == {("grid", "to", "hidden")}
    assert not dynamic.nodes


def test_build_graph_without_nodes_or_stored_graph_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no static nodes"):
        build_graph(DotDict({}), {"grid": True}, "hidden", graph_path=tmp_path / "missing.pt")
