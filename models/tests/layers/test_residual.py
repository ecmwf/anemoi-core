import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.bundle import GraphBundle
from anemoi.models.layers.graph_provider_registry import GraphProviderRegistry
from anemoi.models.layers.residual import SkipConnection
from anemoi.models.layers.residual import TruncatedConnection


@pytest.fixture
def graph_bundle() -> GraphBundle:
    graph = HeteroData()
    graph["data"].num_nodes = 2
    graph["hidden"].num_nodes = 1

    graph[("data", "to", "hidden")].edge_index = torch.tensor([[0, 1], [0, 0]])
    graph[("hidden", "to", "data")].edge_index = torch.tensor([[0, 0], [0, 1]])
    graph[("data", "to", "hidden")].edge_weight = torch.ones(2)
    graph[("hidden", "to", "data")].edge_weight = torch.ones(2)

    return GraphBundle(main=graph, assets={})


def _build_registry(graph_bundle: GraphBundle, edge_weight_attribute: str | None) -> GraphProviderRegistry:
    provider_specs = {
        "down": {
            "_target_": "anemoi.models.layers.graph_provider.ProjectionGraphProvider",
            "edges_name": ["data", "to", "hidden"],
            "edge_weight_attribute": edge_weight_attribute,
            "row_normalize": False,
        },
        "up": {
            "_target_": "anemoi.models.layers.graph_provider.ProjectionGraphProvider",
            "edges_name": ["hidden", "to", "data"],
            "edge_weight_attribute": edge_weight_attribute,
            "row_normalize": False,
        },
    }

    return GraphProviderRegistry({"data": graph_bundle}, provider_specs)


@pytest.fixture
def flat_data():
    return torch.randn(11, 7, 5, 2, 3)  # batch, dates, ensemble, grid, features


def test_truncation_mapper_init(graph_bundle: GraphBundle):
    graph_providers = _build_registry(graph_bundle, "edge_weight")
    _ = TruncatedConnection(
        down_provider="down",
        up_provider="up",
        graph_providers=graph_providers,
        dataset_name="data",
    )


def test_forward(graph_bundle: GraphBundle):
    graph_providers = _build_registry(graph_bundle, "edge_weight")
    mapper = TruncatedConnection(
        down_provider="down",
        up_provider="up",
        graph_providers=graph_providers,
        dataset_name="data",
    )
    x = torch.randn(5, 2, 2, 2, 3)  # (batch, dates, ensemble, grid, features)
    x_truncated = mapper.forward(x)
    assert x_truncated.shape == (5, 2, 2, 3)  # (batch, ensemble, grid, features)


def test_forward_no_weight(graph_bundle: GraphBundle):
    graph_providers = _build_registry(graph_bundle, None)
    mapper = TruncatedConnection(
        down_provider="down",
        up_provider="up",
        graph_providers=graph_providers,
        dataset_name="data",
    )
    x = torch.randn(5, 2, 2, 2, 3)  # (batch, dates, ensemble, grid, features)
    x_truncated = mapper.forward(x)
    assert x_truncated.shape == (5, 2, 2, 3)  # (batch, ensemble, grid, features)


def test_skipconnection(flat_data):
    mapper = SkipConnection()
    out = mapper.forward(flat_data)
    expected_out = flat_data[:, -1, ...]  # Should return the last date

    assert torch.allclose(out, expected_out), "SkipConnection did not return the expected output."
