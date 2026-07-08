# (C) Copyright 2024-2026 Anemoi contributors.
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
from torch_geometric.data import HeteroData

from anemoi.graphs.create import GraphBuilder
from anemoi.graphs.create import GraphCreator
from anemoi.graphs.edges import CutOffEdges
from anemoi.graphs.edges.attributes import EdgeDirection
from anemoi.graphs.edges.attributes import EdgeLength
from anemoi.graphs.nodes import LatLonNodes
from anemoi.graphs.nodes.attributes import SphericalAreaWeights
from anemoi.graphs.nodes.attributes import UniformWeights


def assert_graph_tensors(graph: HeteroData) -> None:
    for nodes in graph.node_stores:
        for node_attr in nodes.node_attrs():
            assert isinstance(nodes[node_attr], torch.Tensor)
            assert nodes[node_attr].dtype in [torch.int32, torch.float32], f"{node_attr} is not int32/float32."

    for edges in graph.edge_stores:
        for edge_attr in edges.edge_attrs():
            assert isinstance(edges[edge_attr], torch.Tensor)
            assert edges[edge_attr].dtype in [torch.int32, torch.float32], f"{edge_attr} is not int32/float32."


class AddProcessedFlag:
    def update_graph(self, graph: HeteroData) -> HeteroData:
        graph["data"].processed = torch.ones((graph["data"].num_nodes, 1), dtype=torch.float32)
        return graph


class TestGraphCreator:

    @pytest.mark.parametrize("name", ["graph.pt", None])
    def test_generate_graph(self, config_file: tuple[Path, str], mock_grids_path: tuple[str, int], name: str):
        """Test GraphCreator workflow."""
        tmp_path, config_name = config_file
        graph_path = tmp_path / name if isinstance(name, str) else None
        config_path = tmp_path / config_name

        graph = GraphCreator(config=config_path).create(save_path=graph_path)

        assert isinstance(graph, HeteroData)
        assert "test_nodes" in graph.node_types
        assert ("test_nodes", "to", "test_nodes") in graph.edge_types

        assert_graph_tensors(graph)

        for nodes in graph.node_stores:
            for node_attr in nodes.node_attrs():
                assert not node_attr.startswith("_")
        for edges in graph.edge_stores:
            for edge_attr in edges.edge_attrs():
                assert not edge_attr.startswith("_")

        if graph_path is not None:
            assert graph_path.exists()
            graph_saved = torch.load(graph_path, weights_only=False)
            assert graph.node_types == graph_saved.node_types
            assert graph.edge_types == graph_saved.edge_types

    @pytest.mark.parametrize("name", ["graph.pt", None])
    def test_generate_graph_from_python_api(self, tmp_path: Path, name: str):
        """Test graph creation directly from Python builders."""
        graph_path = tmp_path / name if isinstance(name, str) else None

        data_nodes = LatLonNodes(
            latitudes=[45.0, 45.0, 40.0, 40.0],
            longitudes=[5.0, 10.0, 10.0, 5.0],
            name="data",
            attributes=[UniformWeights(name="uniform_weights")],
        )
        hidden_nodes = LatLonNodes(
            latitudes=[45.0, 45.0, 40.0, 40.0],
            longitudes=[5.0, 10.0, 10.0, 5.0],
            name="hidden",
            attributes=[UniformWeights(name="uniform_weights")],
        )

        graph = GraphBuilder(
            nodes=[data_nodes, hidden_nodes],
            edges=[
                CutOffEdges(
                    source_name="data",
                    target_name="hidden",
                    cutoff_distance_km=20000,
                    attributes=[EdgeLength(norm="unit-std"), EdgeDirection(norm="unit-std")],
                ),
            ],
        ).create(save_path=graph_path)

        assert isinstance(graph, HeteroData)
        assert {"data", "hidden"}.issubset(set(graph.node_types))
        assert ("data", "to", "hidden") in graph.edge_types

        assert graph["data"].x.shape == (4, 2)
        assert graph["hidden"].x.shape == (4, 2)
        assert graph[("data", "to", "hidden")].edge_index.shape[0] == 2
        assert "uniform_weights" in graph["data"]
        assert "edge_length" in graph[("data", "to", "hidden")]
        assert "edge_direction" in graph[("data", "to", "hidden")]

        assert_graph_tensors(graph)

        for nodes in graph.node_stores:
            for node_attr in nodes.node_attrs():
                assert not node_attr.startswith("_")
        for edges in graph.edge_stores:
            for edge_attr in edges.edge_attrs():
                assert not edge_attr.startswith("_")

        if graph_path is not None:
            assert graph_path.exists()
            graph_saved = torch.load(graph_path, weights_only=False)
            assert graph.node_types == graph_saved.node_types
            assert graph.edge_types == graph_saved.edge_types

    def test_post_processors_are_applied(self):
        """Test that Python API post-processors are applied during graph creation."""
        graph = GraphBuilder(
            nodes=[
                LatLonNodes(
                    latitudes=[45.0, 45.0],
                    longitudes=[5.0, 10.0],
                    name="data",
                )
            ],
            post_processors=[AddProcessedFlag()],
        ).create()

        assert "processed" in graph["data"]
        assert torch.equal(graph["data"].processed, torch.ones((2, 1), dtype=torch.float32))
