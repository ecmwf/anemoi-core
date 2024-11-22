import numpy as np
import torch

from anemoi.graphs.utils import concat_edges


def test_concat_edges():
    edge_indices1 = torch.tensor([[0, 1, 2, 3], [-1, -2, -3, -4]], dtype=torch.int64)
    edge_indices2 = torch.tensor(np.array([[0, 4], [-1, -5]]), dtype=torch.int64)
    no_edges = torch.tensor([[], []], dtype=torch.int64)

    result1 = concat_edges(edge_indices1, edge_indices2)
    result2 = concat_edges(no_edges, edge_indices2)

    expected1 = torch.tensor([[0, 1, 2, 3, 4], [-1, -2, -3, -4, -5]], dtype=torch.int64)

    assert torch.allclose(result1, expected1)
    assert torch.allclose(result2, edge_indices2)
