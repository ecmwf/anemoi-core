import pytest
from torch_geometric.data import HeteroData

from anemoi.graphs.edges import MultiScaleEdges
from anemoi.graphs.nodes import HexNodes
from anemoi.graphs.nodes import TriNodes


class TestMultiScaleEdgesInit:
    def test_init(self):
        """Test MultiScaleEdges initialization."""
        assert isinstance(MultiScaleEdges("test_nodes", "test_nodes", 1), MultiScaleEdges)

    @pytest.mark.parametrize("x_hops", [-0.5, "hello", None, -4])
    def test_fail_init(self, x_hops: str):
        """Test MultiScaleEdges initialization with invalid x_hops."""
        with pytest.raises(AssertionError):
            MultiScaleEdges("test_nodes", "test_nodes", x_hops)

    def test_fail_init_diff_nodes(self):
        """Test MultiScaleEdges initialization with invalid nodes."""
        with pytest.raises(AssertionError):
            MultiScaleEdges("test_nodes", "test_nodes2", 0)


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

        edges = MultiScaleEdges("test_tri_nodes", "test_tri_nodes", 1)
        graph = edges.update_graph(tri_ico_graph)
        assert ("test_tri_nodes", "to", "test_tri_nodes") in graph.edge_types

    def test_transform_same_src_dst_hex_nodes(self, hex_ico_graph: HeteroData):
        """Test MultiScaleEdges update method."""

        edges = MultiScaleEdges("test_hex_nodes", "test_hex_nodes", 1)
        graph = edges.update_graph(hex_ico_graph)
        assert ("test_hex_nodes", "to", "test_hex_nodes") in graph.edge_types

    def test_transform_fail_nodes(self, tri_ico_graph: HeteroData):
        """Test MultiScaleEdges update method with wrong node type."""
        edges = MultiScaleEdges("fail_nodes", "fail_nodes", 1)
        with pytest.raises(AssertionError):
            edges.update_graph(tri_ico_graph)
