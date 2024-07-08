import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.nodes.attributes import AreaWeights
from anemoi.graphs.nodes.attributes import UniformWeights


@pytest.mark.parametrize("norm", [None, "l1", "l2", "unit-max", "unit-std"])
def test_uniform_weights(graph_with_nodes: HeteroData, norm: str):
    """Test attribute builder for UniformWeights."""
    node_attr_builder = UniformWeights(norm=norm)
    weights = node_attr_builder.compute(graph_with_nodes, "test_nodes")

    assert weights is not None
    assert isinstance(weights, torch.Tensor)
    assert weights.shape[0] == graph_with_nodes["test_nodes"].x.shape[0]


@pytest.mark.parametrize("norm", ["l3", "invalide"])
def test_uniform_weights_fail(graph_with_nodes: HeteroData, norm: str):
    """Test attribute builder for UniformWeights with invalid norm."""
    with pytest.raises(ValueError):
        node_attr_builder = UniformWeights(norm=norm)
        node_attr_builder.compute(graph_with_nodes, "test_nodes")


def test_area_weights(graph_with_nodes: HeteroData):
    """Test attribute builder for AreaWeights."""
    node_attr_builder = AreaWeights()
    weights = node_attr_builder.compute(graph_with_nodes, "test_nodes")

    assert weights is not None
    assert isinstance(weights, torch.Tensor)
    assert weights.shape[0] == graph_with_nodes["test_nodes"].x.shape[0]


@pytest.mark.parametrize("radius", [-1.0, "hello", None])
def test_area_weights_fail(graph_with_nodes: HeteroData, radius: float):
    """Test attribute builder for AreaWeights with invalid radius."""
    with pytest.raises(ValueError):
        node_attr_builder = AreaWeights(radius=radius)
        node_attr_builder.compute(graph_with_nodes, "test_nodes")
