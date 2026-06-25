import numpy as np
import torch

from anemoi.graphs.utils import concat_edges
from anemoi.graphs.utils import get_edge_attributes
from anemoi.graphs.utils import intersect_edges


def test_concat_edges():
    edge_indices1 = torch.tensor([[0, 1, 2, 3], [-1, -2, -3, -4]], dtype=torch.int64)
    edge_indices2 = torch.tensor(np.array([[0, 4], [-1, -5]]), dtype=torch.int64)
    no_edges = torch.tensor([[], []], dtype=torch.int64)

    result1 = concat_edges(edge_indices1, edge_indices2)
    result2 = concat_edges(no_edges, edge_indices2)

    expected1 = torch.tensor([[0, 1, 2, 3, 4], [-1, -2, -3, -4, -5]], dtype=torch.int64)

    assert torch.allclose(result1, expected1)
    assert torch.allclose(result2, edge_indices2)


def test_intersect_edges():
    edge_indices1 = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int64)
    edge_indices2 = torch.tensor([[1, 3, 5], [5, 7, 9]], dtype=torch.int64)
    no_edges = torch.tensor([[], []], dtype=torch.int64)

    # (1,5) and (3,7) are the only columns present in both inputs.
    result = intersect_edges(edge_indices1, edge_indices2)
    expected = torch.tensor([[1, 3], [5, 7]], dtype=torch.int64)

    assert torch.equal(result, expected)
    assert intersect_edges(no_edges, edge_indices2).shape == (2, 0)
    assert intersect_edges(no_edges, edge_indices2).dtype == torch.int64
    assert intersect_edges(edge_indices1, no_edges).shape == (2, 0)
    assert intersect_edges(edge_indices1, no_edges).dtype == torch.int64

    # Non-empty inputs with no shared columns — exercises the all-False isin mask path.
    no_overlap1 = torch.tensor([[0], [1]], dtype=torch.int64)
    no_overlap2 = torch.tensor([[2], [3]], dtype=torch.int64)
    assert intersect_edges(no_overlap1, no_overlap2).shape == (2, 0)


def test_get_edge_attributes():
    mock_config = {
        "nodes": "mock_nodes_config",
        "edges": [
            {
                "source_name": "mock_nodes",
                "target_name": "mock_nodes",
                "attributes": {"attr": "attr_config"},
                "extra_keys": "extra_values",
            },
        ],
    }

    # test non-empty selection
    edge_attrs = get_edge_attributes(mock_config, "mock_nodes", "mock_nodes")
    expected = mock_config["edges"][0]["attributes"]
    assert edge_attrs == expected

    # test empty selection
    edge_attrs = get_edge_attributes(mock_config, "mock_nodes", "other_nodes")
    assert edge_attrs == {}

    edge_attrs = get_edge_attributes(mock_config, "other_nodes", "mock_nodes")
    assert edge_attrs == {}
