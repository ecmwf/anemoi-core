import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.nodes import HexNodes
from anemoi.graphs.nodes.builder import BaseNodeBuilder


@pytest.mark.parametrize("resolution", [0, 2])
def test_init(resolution: list[int]):
    """Test TrirefinedIcosahedralNodes initialization."""

    node_builder = HexNodes(resolution, "test_nodes")
    assert isinstance(node_builder, BaseNodeBuilder)
    assert isinstance(node_builder, HexNodes)


def test_get_coordinates():
    """Test get_coordinates method."""
    node_builder = HexNodes(0, "test_nodes")
    coords = node_builder.get_coordinates()
    assert isinstance(coords, torch.Tensor)
    assert coords.shape == (122, 2)


def test_update_graph():
    """Test update_graph method."""
    node_builder = HexNodes(0, "test_nodes")
    graph = HeteroData()
    graph = node_builder.update_graph(graph, {})
    assert "_resolutions" in graph["test_nodes"]
    assert "_nx_graph" in graph["test_nodes"]
    assert "_node_ordering" in graph["test_nodes"]
    assert len(graph["test_nodes"]["_node_ordering"]) == graph["test_nodes"].num_nodes
