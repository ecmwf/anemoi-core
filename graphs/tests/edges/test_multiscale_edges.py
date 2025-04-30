# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest
from torch_geometric.data import HeteroData

from anemoi.graphs.edges import MultiScaleEdges
from anemoi.graphs.generate.masks import KNNAreaMaskBuilder
from anemoi.graphs.nodes import HexNodes
from anemoi.graphs.nodes import StretchedTriNodes
from anemoi.graphs.nodes import TriNodes


class TestMultiScaleEdgesInit:
    def test_init(self):
        """Test MultiScaleEdges initialization."""
        assert isinstance(MultiScaleEdges("test_nodes", "test_nodes", 1, None), MultiScaleEdges)
        assert isinstance(MultiScaleEdges("test_nodes", "test_nodes", 1, scale_resolutions=4), MultiScaleEdges)
        assert isinstance(MultiScaleEdges("test_nodes", "test_nodes", 1, scale_resolutions=[1, 2, 3]), MultiScaleEdges)

    @pytest.mark.parametrize("x_hops", [-0.5, "hello", None, -4])
    def test_fail_init_invalid_x_hops(self, x_hops: str):
        """Test MultiScaleEdges initialization with invalid x_hops."""
        with pytest.raises(AssertionError):
            MultiScaleEdges("test_nodes", "test_nodes", x_hops, None)

    @pytest.mark.parametrize("scale_resolutions", [0, -1, [0], [-1], "invalid"])
    def test_fail_init_invalid_scale_resolutions(self, scale_resolutions):
        """Test MultiScaleEdges initialization with invalid scale_resolutions."""
        with pytest.raises(AssertionError):
            MultiScaleEdges("test_nodes", "test_nodes", 1, scale_resolutions=scale_resolutions)

    def test_fail_init_diff_nodes(self):
        """Test MultiScaleEdges initialization with invalid nodes."""
        with pytest.raises(AssertionError):
            MultiScaleEdges("test_nodes", "test_nodes2", 0, None)


class TestMultiScaleEdgesTransform:

    @pytest.fixture()
    def tri_ico_graph(self) -> HeteroData:
        """Return a HeteroData object with MultiScaleEdges."""
        graph = HeteroData()
        graph = TriNodes(1, "test_tri_nodes").update_graph(graph, {})
        graph["fail_nodes"].x = [1, 2, 3]
        graph["fail_nodes"].node_type = "FailNodes"
        return graph

    @pytest.fixture()
    def hex_ico_graph(self) -> HeteroData:
        """Return a HeteroData object with TriNodes."""
        graph = HeteroData()
        graph = HexNodes(1, "test_hex_nodes").update_graph(graph, {})
        graph["fail_nodes"].x = [1, 2, 3]
        graph["fail_nodes"].node_type = "FailNodes"
        return graph

    def test_transform_same_src_dst_tri_nodes(self, tri_ico_graph: HeteroData):
        """Test MultiScaleEdges update method."""

        edges = MultiScaleEdges("test_tri_nodes", "test_tri_nodes", 1, None)
        graph = edges.update_graph(tri_ico_graph)
        assert ("test_tri_nodes", "to", "test_tri_nodes") in graph.edge_types

    def test_transform_same_src_dst_hex_nodes(self, hex_ico_graph: HeteroData):
        """Test MultiScaleEdges update method."""

        edges = MultiScaleEdges("test_hex_nodes", "test_hex_nodes", 1, None)
        graph = edges.update_graph(hex_ico_graph)
        assert ("test_hex_nodes", "to", "test_hex_nodes") in graph.edge_types

    @pytest.mark.parametrize("scale_resolutions", [1, [1], [1, 2], None])
    def test_transform_with_scale_resolutions(self, tri_ico_graph: HeteroData, scale_resolutions):
        """Test MultiScaleEdges with different scale_resolutions configurations."""
        edges = MultiScaleEdges("test_tri_nodes", "test_tri_nodes", 1, scale_resolutions=scale_resolutions)
        graph = edges.update_graph(tri_ico_graph)

        assert ("test_tri_nodes", "to", "test_tri_nodes") in graph.edge_types
        assert len(graph[("test_tri_nodes", "to", "test_tri_nodes")].edge_index) > 0
        assert graph[("test_tri_nodes", "to", "test_tri_nodes")].edge_index.dim() == 2

    def test_transform_fail_nodes(self, tri_ico_graph: HeteroData):
        """Test MultiScaleEdges update method with wrong node type."""
        edges = MultiScaleEdges("fail_nodes", "fail_nodes", 1, None)
        with pytest.raises(ValueError):
            edges.update_graph(tri_ico_graph)


class TestMultiScaleEdgesStretched:

    @pytest.fixture()
    def tri_graph(self, mocker) -> HeteroData:
        """Return a HeteroData object with stretched Tri nodes."""
        graph = HeteroData()
        node_builder = StretchedTriNodes(5, 7, "hidden", None, None)
        node_builder.area_mask_builder = KNNAreaMaskBuilder("hidden", 400)
        node_builder.area_mask_builder.fit_coords(np.array([[0, 0]]))
        # We are considering a 400km radius circle centered at (0, 0) as the area of
        # interest for the stretched graph.

        mocker.patch.object(node_builder.area_mask_builder, "fit", return_value=None)

        graph = node_builder.update_graph(graph, {})
        return graph

    def test_edges(self, tri_graph: HeteroData):
        """Test MultiScaleEdges update method."""
        edges = MultiScaleEdges("hidden", "hidden", x_hops=1, scale_resolutions=None)
        graph = edges.update_graph(tri_graph)
        assert ("hidden", "to", "hidden") in graph.edge_types
        assert len(graph[("hidden", "to", "hidden")].edge_index) > 0
