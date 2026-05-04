import os
import tempfile

import numpy as np
import pytest
import scipy.sparse
import torch
from torch_geometric.data import HeteroData

from anemoi.models.layers.residual import InterpolationConnection
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


def test_skipconnection_step_index():
    """SkipConnection with step=0 returns the first timestep."""
    x = torch.randn(2, 3, 1, 10, 4)
    mapper = SkipConnection(step=0)
    out = mapper(x)
    assert torch.equal(out, x[:, 0, ...])


def test_skipconnection_expand_time():
    """SkipConnection with n_step_output expands the time dimension."""
    x = torch.randn(2, 3, 1, 10, 4)
    mapper = SkipConnection(step=-1)
    out = mapper(x, n_step_output=5)
    assert out.shape == (2, 5, 1, 10, 4)
    # All timesteps should be identical (expanded, not repeated)
    for t in range(5):
        assert torch.equal(out[:, t], out[:, 0])


@pytest.fixture
def interpolation_npz():
    """Create a temporary .npz file with a simple sparse interpolation matrix."""
    # 3 source nodes -> 5 target nodes
    rows = np.array([0, 1, 2, 3, 4])
    cols = np.array([0, 0, 1, 1, 2])
    data = np.array([1.0, 0.5, 0.5, 0.5, 0.5])
    sparse_matrix = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(5, 3))
    fd, path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    scipy.sparse.save_npz(path, sparse_matrix)
    yield path
    os.unlink(path)


def test_interpolation_connection_forward(interpolation_npz):
    """InterpolationConnection upsamples from source to target grid."""
    conn = InterpolationConnection(interpolation_file_path=interpolation_npz, step=-1)
    # x: (batch=2, time=1, ensemble=1, grid=3, features=4)
    x = torch.ones(2, 1, 1, 3, 4)
    out = conn(x)
    # Output should have 5 grid points (target grid)
    assert out.shape == (2, 1, 5, 4)


def test_interpolation_connection_preserves_batch(interpolation_npz):
    """InterpolationConnection correctly handles different batch sizes."""
    conn = InterpolationConnection(interpolation_file_path=interpolation_npz, step=-1)
    for batch_size in [1, 4, 8]:
        x = torch.randn(batch_size, 1, 1, 3, 4)
        out = conn(x)
        assert out.shape[0] == batch_size
