import numpy as np
import torch

from anemoi.graphs.utils import concat_edges
from anemoi.graphs.utils import get_edge_attributes


def test_concat_edges():
    edge_indices1 = torch.tensor([[0, 1, 2, 3], [-1, -2, -3, -4]], dtype=torch.int64)
    edge_indices2 = torch.tensor(np.array([[0, 4], [-1, -5]]), dtype=torch.int64)
    no_edges = torch.tensor([[], []], dtype=torch.int64)

    result1 = concat_edges(edge_indices1, edge_indices2)
    result2 = concat_edges(no_edges, edge_indices2)

    expected1 = torch.tensor([[0, 1, 2, 3, 4], [-1, -2, -3, -4, -5]], dtype=torch.int64)

    assert torch.allclose(result1, expected1)
    assert torch.allclose(result2, edge_indices2)

def test_get_edge_attributes():
    mock_config = {
        "nodes": "mock_nodes_config",
        "edges": [
            {
                "source_name": "mock_nodes",
                "target_name": "mock_nodes",
                "attributes" : {'attr': 'attr_config'},
                "extra_keys": "extra_values",
            },
        ],
    }
    edge_attrs1=get_edge_attributes(mock_config, "mock_nodes", "mock_nodes")
    edge_attrs2=get_edge_attributes(mock_config, "mock_nodes", "other_nodes")
    edge_attrs3=get_edge_attributes(mock_config, "other_nodes", "mock_nodes")

    expected1=mock_config["edges"]["attributes"]

    assert edge_attrs1 == expected1
    assert edge_attrs2 == {}
    assert edge_attrs3 == {}

