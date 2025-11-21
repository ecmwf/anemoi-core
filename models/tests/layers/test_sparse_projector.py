from anemoi.models.layers.sparse_projector import SparseProjector
import torch
import pytest


@pytest.fixture
def edge_index():
    return torch.tensor([[0, 1, 0, 2, 3], [0, 0, 1, 1, 1]])


def test_sparse_projector(edge_index):
    projector = SparseProjector(edge_index=edge_index, weights=torch.ones(edge_index.size(1)), src_size=4, dst_size=2)

    # test forward, input.shape = (bs, num_src_nodes, feature_dim)
    src_features = torch.tensor([[[1.0], [2.0], [3.0], [5.0]]])  # shape (1, 4, 1)
    out = projector(src_features)
    assert out.shape == (1, 2, 1)
    expected_out = torch.tensor([[[1.5], [3]]])
    assert torch.allclose(out, expected_out), f"Expected {expected_out}, but got {out}."


def test_fail_no_weights(edge_index):
    with pytest.raises(ValueError, match="Weights must be provided for SparseProjector."):
        SparseProjector(edge_index=edge_index, src_size=4, dst_size=2)
