# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from torch_geometric.data import HeteroData

from anemoi.graphs.edges import HEALPixMultiScaleEdges
from anemoi.graphs.nodes import HEALPixNodes


class TestMultiScaleEdgesInit:
    def test_init(self):
        """Test MultiScaleEdges initialisation."""
        assert isinstance(HEALPixMultiScaleEdges("test_nodes", "test_nodes", None), HEALPixMultiScaleEdges)
        assert isinstance(
            HEALPixMultiScaleEdges("test_nodes", "test_nodes", scale_resolutions=4), HEALPixMultiScaleEdges
        )
        assert isinstance(
            HEALPixMultiScaleEdges("test_nodes", "test_nodes", scale_resolutions=[1, 2, 3]), HEALPixMultiScaleEdges
        )

    @pytest.mark.parametrize("scale_resolutions", [0, -1, [0], [-1], "invalid"])
    def test_fail_init_invalid_scale_resolutions(self, scale_resolutions):
        """Test MultiScaleEdges initialisation with invalid scale_resolutions."""
        with pytest.raises(AssertionError):
            HEALPixMultiScaleEdges("test_nodes", "test_nodes", scale_resolutions=scale_resolutions)

    def test_fail_init_diff_nodes(self):
        """Test MultiScaleEdges initialisation with invalid nodes."""
        with pytest.raises(AssertionError):
            HEALPixMultiScaleEdges("test_nodes", "test_nodes2", None)


class TestHEALPixMultiScaleEdgesTransform:

    @pytest.fixture()
    def healpix_graph(self) -> HeteroData:
        """Return a HeteroData object with HEALPixMultiScaleEdges."""
        graph = HeteroData()
        graph = HEALPixNodes(1, "test_tri_nodes").update_graph(graph, {})
        graph["fail_nodes"].x = [1, 2, 3]
        graph["fail_nodes"].node_type = "FailNodes"
        return graph

    def test_transform_same_src_dst_tri_nodes(self, healpix_graph: HeteroData):
        """Test HEALPixMultiScaleEdges update method."""

        edges = HEALPixMultiScaleEdges("test_tri_nodes", "test_tri_nodes", None)
        graph = edges.update_graph(healpix_graph)
        assert ("test_tri_nodes", "to", "test_tri_nodes") in graph.edge_types

    @pytest.mark.parametrize("scale_resolutions", [1, [1], [1, 2], None])
    def test_transform_with_scale_resolutions(self, healpix_graph: HeteroData, scale_resolutions):
        """Test HEALPixMultiScaleEdges with different scale_resolutions configurations."""
        edges = HEALPixMultiScaleEdges("test_tri_nodes", "test_tri_nodes", scale_resolutions=scale_resolutions)
        graph = edges.update_graph(healpix_graph)

        assert ("test_tri_nodes", "to", "test_tri_nodes") in graph.edge_types
        assert len(graph[("test_tri_nodes", "to", "test_tri_nodes")].edge_index) > 0
        assert graph[("test_tri_nodes", "to", "test_tri_nodes")].edge_index.dim() == 2

    def test_transform_fail_nodes(self, healpix_graph: HeteroData):
        """Test MultiScaleEdges update method with wrong node type."""
        edges = HEALPixMultiScaleEdges("fail_nodes", "fail_nodes", None)
        with pytest.raises(AssertionError):
            edges.update_graph(healpix_graph)
