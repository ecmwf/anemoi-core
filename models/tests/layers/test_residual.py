import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.models.layers.residual import SkipConnection
from anemoi.models.layers.residual import TruncatedConnection


@pytest.fixture
def graph_data():
    g = HeteroData()
    g["data"].num_nodes = 2
    g["hidden"].num_nodes = 1
    g["hidden", "to", "data"].edge_index = torch.tensor([[0, 0], [0, 1]])
    g["hidden", "to", "data"].edge_length = torch.tensor([1.0, 2.0])
    g["data", "to", "hidden"].edge_index = torch.tensor([[0, 1], [0, 0]])
    g["data", "to", "hidden"].edge_length = torch.tensor([1.0, 2.0])
    g["data"].weight = torch.tensor([1.0, 0.5])  # Example weights for data nodes
    g["hidden"].weight = torch.tensor([0.8])  # Example weight for hidden node
    return g


@pytest.fixture
def flat_data():
    x = torch.randn(11, 7, 5, 2, 3)  # batch, dates, ensemble, grid, features
    return x


@pytest.fixture
def edge_index():
    return torch.tensor([[0, 1, 1], [1, 0, 2]])


def test_truncation_mapper_init(graph_data):
    _ = TruncatedConnection(
        graph_data, data_nodes="data", truncation_nodes="hidden", edge_weight_attribute="edge_length"
    )


def test_forward(graph_data):
    mapper = TruncatedConnection(
        graph_data, data_nodes="data", truncation_nodes="hidden", edge_weight_attribute="edge_length"
    )
    x = torch.randn(5, 2, 2, 2, 3)  # (batch, dates, ensemble, grid, features)
    x_truncated = mapper.forward(x)
    assert x_truncated.shape == (5, 2, 2, 3)  # (batch, ensemble, coarse_grid, features)


def test_forward_no_weight(graph_data):
    mapper = TruncatedConnection(graph_data, data_nodes="data", truncation_nodes="hidden")
    x = torch.randn(5, 2, 2, 2, 3)  # (batch, dates, ensemble, grid, features)
    x_truncated = mapper.forward(x)
    assert x_truncated.shape == (5, 2, 2, 3)  # (batch, ensemble, coarse_grid, features)


def test_forward_with_src_node_weight(graph_data):
    mapper = TruncatedConnection(
        graph_data, data_nodes="data", truncation_nodes="hidden", src_node_weight_attribute="weight"
    )
    x = torch.randn(5, 2, 2, 2, 3)  # (batch, dates, ensemble, grid, features)
    x_truncated = mapper.forward(x)
    assert x_truncated.shape == (5, 2, 2, 3)  # (batch, ensemble, coarse_grid, features)


def test_skipconnection(flat_data):
    mapper = SkipConnection()
    out = mapper.forward(flat_data)
    expected_out = flat_data[:, -1, ...]  # Should return the last date

    assert torch.allclose(out, expected_out), "SkipConnection did not return the expected output."
