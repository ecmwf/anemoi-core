import pytest
import torch
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.layers.residual import SkipConnection
from anemoi.models.layers.residual import TruncatedConnection


@pytest.fixture
def data_indices():
    config = DictConfig(
        {
            "data": {
                "forcing": [],
                "diagnostic": [],
            },
        },
    )
    # All variables are prognostic and present in model.input with the same indices.
    name_to_index = {"a": 0, "b": 1, "c": 2}
    return IndexCollection(config.data, name_to_index=name_to_index)


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


def test_truncation_mapper_init(graph_data, data_indices):
    _ = TruncatedConnection(
        graph_data,
        data_nodes="data",
        truncation_nodes="hidden",
        edge_weight_attribute="edge_length",
        data_indices=data_indices,
    )


def test_forward(graph_data, data_indices):
    mapper = TruncatedConnection(
        graph_data,
        data_nodes="data",
        truncation_nodes="hidden",
        edge_weight_attribute="edge_length",
        data_indices=data_indices,
    )
    x = torch.randn(5, 2, 2, 2, 3)  # (batch, dates, ensemble, grid, features)
    x_truncated = mapper.forward(x)
    assert x_truncated.shape == (5, 2, 2, 3)  # (batch, ensemble, coarse_grid, features)


def test_forward_no_weight(graph_data, data_indices):
    mapper = TruncatedConnection(graph_data, data_nodes="data", truncation_nodes="hidden", data_indices=data_indices)
    x = torch.randn(5, 2, 2, 2, 3)  # (batch, dates, ensemble, grid, features)
    x_truncated = mapper.forward(x)
    assert x_truncated.shape == (5, 2, 2, 3)  # (batch, ensemble, coarse_grid, features)


def test_forward_with_src_node_weight(graph_data, data_indices):
    mapper = TruncatedConnection(
        graph_data,
        data_nodes="data",
        truncation_nodes="hidden",
        src_node_weight_attribute="weight",
        data_indices=data_indices,
    )
    x = torch.randn(5, 2, 2, 2, 3)  # (batch, dates, ensemble, grid, features)
    x_truncated = mapper.forward(x)
    assert x_truncated.shape == (5, 2, 2, 3)  # (batch, ensemble, coarse_grid, features)


def test_skipconnection(flat_data, data_indices):
    mapper = SkipConnection(data_indices=data_indices)
    out = mapper.forward(flat_data)
    expected_out = flat_data[:, -1, ...]  # Should return the last date

    assert torch.allclose(out, expected_out), "SkipConnection did not return the expected output."


def test_skipconnection_drop(flat_data, data_indices):
    """Variables listed under ``drop`` are zeroed out in the skip branch."""
    drop = ["a", "c"]
    mapper = SkipConnection(drop=drop, data_indices=data_indices)

    drop_indices = [data_indices.model.input.name_to_index[name] for name in drop]
    keep_indices = [i for i in range(flat_data.shape[-1]) if i not in drop_indices]

    out = mapper.forward(flat_data.clone())

    expected_kept = flat_data[:, -1, ..., keep_indices]
    assert torch.allclose(out[..., keep_indices], expected_kept), "Non-dropped variables were modified."
    assert torch.all(out[..., drop_indices] == 0.0), "Dropped variables should be zeroed out."


def test_skipconnection_drop_invalid_variable(data_indices):
    """Dropping a variable not present in model.input must fail loudly."""
    with pytest.raises(AssertionError):
        SkipConnection(drop=["not_a_variable"], data_indices=data_indices)


def test_skipconnection_drop_default_keeps_all(flat_data, data_indices):
    """By default, no variables are dropped from the skip connection."""
    mapper = SkipConnection(data_indices=data_indices)
    assert mapper.drop_indices == []
    out = mapper.forward(flat_data.clone())
    assert torch.allclose(out, flat_data[:, -1, ...])
