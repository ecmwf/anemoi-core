import pytest
import torch
from omegaconf import OmegaConf

from anemoi.models.layers.residual import SkipConnection
from anemoi.models.layers.residual import TruncatedConnection


@pytest.fixture
def truncation_graph():
    graph_config = OmegaConf.create(
        {
            "nodes": {
                "data": {
                    "node_builder": {
                        "_target_": "anemoi.graphs.nodes.LatLonNodes",
                        "latitudes": [0.0, 1.0],
                        "longitudes": [0.0, 1.0],
                    },
                },
                "hidden": {
                    "node_builder": {
                        "_target_": "anemoi.graphs.nodes.LatLonNodes",
                        "latitudes": [0.0],
                        "longitudes": [0.0],
                    },
                },
            },
            "edges": [
                {
                    "source_name": "data",
                    "target_name": "hidden",
                    "edge_builders": [
                        {
                            "_target_": "anemoi.graphs.edges.CutOffEdges",
                            "cutoff_distance_km": 10000,
                            "max_num_neighbours": 2,
                        },
                    ],
                    "attributes": {
                        "edge_length": {
                            "_target_": "anemoi.graphs.edges.attributes.EdgeLength",
                            "norm": "l1",
                        },
                    },
                },
                {
                    "source_name": "hidden",
                    "target_name": "data",
                    "edge_builders": [
                        {
                            "_target_": "anemoi.graphs.edges.CutOffEdges",
                            "cutoff_distance_km": 10000,
                            "max_num_neighbours": 2,
                        },
                    ],
                    "attributes": {
                        "edge_length": {
                            "_target_": "anemoi.graphs.edges.attributes.EdgeLength",
                            "norm": "l1",
                        },
                    },
                },
            ],
            "post_processors": [],
        },
    )

    return {
        "graph_config": graph_config,
        "down_edges_name": ["data", "to", "hidden"],
        "up_edges_name": ["hidden", "to", "data"],
        "edge_weight_attribute": "edge_length",
    }


@pytest.fixture
def flat_data():
    return torch.randn(11, 7, 5, 2, 3)  # batch, dates, ensemble, grid, features


def test_truncation_mapper_init(truncation_graph):
    _ = TruncatedConnection(truncation_graph=truncation_graph)


def test_forward(truncation_graph):
    mapper = TruncatedConnection(truncation_graph=truncation_graph)
    x = torch.randn(5, 2, 2, 2, 3)  # (batch, dates, ensemble, grid, features)
    x_truncated = mapper.forward(x)
    assert x_truncated.shape == (5, 2, 2, 3)  # (batch, ensemble, grid, features)


def test_forward_no_weight(truncation_graph):
    graph_spec = {**truncation_graph, "edge_weight_attribute": None}
    mapper = TruncatedConnection(truncation_graph=graph_spec)
    x = torch.randn(5, 2, 2, 2, 3)  # (batch, dates, ensemble, grid, features)
    x_truncated = mapper.forward(x)
    assert x_truncated.shape == (5, 2, 2, 3)  # (batch, ensemble, grid, features)


def test_skipconnection(flat_data):
    mapper = SkipConnection()
    out = mapper.forward(flat_data)
    expected_out = flat_data[:, -1, ...]  # Should return the last date

    assert torch.allclose(out, expected_out), "SkipConnection did not return the expected output."
