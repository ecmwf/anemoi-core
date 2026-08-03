# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.edges.attributes import EdgeLength
from anemoi.graphs.edges.attributes import Timedeltas
from anemoi.models.layers.graph_provider import DynamicGraphProvider
from anemoi.models.layers.graph_provider import ProjectionGraphProvider


def test_projection_graph_provider_preserves_row_normalized_weights() -> None:
    graph = HeteroData()
    graph["src"].num_nodes = 3
    graph["dst"].num_nodes = 2

    edge_index = torch.tensor([[0, 1, 2, 0], [0, 0, 1, 1]])
    edge_weight = torch.tensor([0.25, 0.75, 0.6, 0.4])  # per-target sums: [1.0, 1.0]

    graph[("src", "to", "dst")].edge_index = edge_index
    graph[("src", "to", "dst")].gauss_weight = edge_weight

    provider = ProjectionGraphProvider(
        graph=graph,
        edges_name=("src", "to", "dst"),
        edge_weight_attribute="gauss_weight",
        row_normalize=False,
    )

    edges = provider.get_edges()
    assert edges.layout == torch.sparse_csr

    matrix = edges.to_dense()
    assert matrix.shape == (graph["dst"].num_nodes, graph["src"].num_nodes)

    row_sums = matrix.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_projection_graph_provider_accepts_int32_edge_index() -> None:
    graph = HeteroData()
    graph["src"].num_nodes = 3
    graph["dst"].num_nodes = 2

    # GraphCreator may yield int32 edge indices; provider should handle this.
    edge_index = torch.tensor([[0, 1, 2, 0], [0, 0, 1, 1]], dtype=torch.int32)
    edge_weight = torch.tensor([0.25, 0.75, 0.6, 0.4], dtype=torch.float32)

    graph[("src", "to", "dst")].edge_index = edge_index
    graph[("src", "to", "dst")].gauss_weight = edge_weight

    provider = ProjectionGraphProvider(
        graph=graph,
        edges_name=("src", "to", "dst"),
        edge_weight_attribute="gauss_weight",
        row_normalize=False,
    )

    edges = provider.get_edges()
    assert edges.layout == torch.sparse_csr

    matrix = edges.to_dense()
    assert matrix.shape == (graph["dst"].num_nodes, graph["src"].num_nodes)
    row_sums = matrix.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def _make_graph_with_edges() -> HeteroData:
    graph = HeteroData()
    graph["data"].num_nodes = 3
    graph["target"].num_nodes = 2
    edge_index = torch.tensor([[0, 1, 2, 0], [0, 0, 1, 1]])
    edge_weight = torch.tensor([0.25, 0.75, 0.6, 0.4])
    graph[("data", "to", "target")].edge_index = edge_index
    graph[("data", "to", "target")].gauss_weight = edge_weight
    return graph


def test_from_config_returns_none_for_none() -> None:
    assert ProjectionGraphProvider.from_config(None) is None


def test_from_config_returns_none_for_empty_dict() -> None:
    assert ProjectionGraphProvider.from_config({}) is None


def test_from_config_file_mode(mocker) -> None:
    import numpy as np

    # rows do not sum to 1, so row_normalize (forwarded to the file path) is observable.
    mocker.patch(
        "anemoi.models.layers.graph_provider.load_npz",
        return_value=csr_matrix(np.array([[2.0, 2.0], [1.0, 3.0]])),
    )

    normalized = ProjectionGraphProvider.from_config({"matrix_path": "/fake/path.npz", "row_normalize": True})
    assert isinstance(normalized, ProjectionGraphProvider)
    row_sums = normalized.get_edges().to_dense().sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    unnormalized = ProjectionGraphProvider.from_config({"matrix_path": "/fake/path.npz", "row_normalize": False})
    row_sums = unnormalized.get_edges().to_dense().sum(dim=1)
    assert torch.allclose(row_sums, torch.tensor([4.0, 4.0]), atol=1e-6)


def test_from_config_edges_mode() -> None:
    graph = _make_graph_with_edges()
    provider = ProjectionGraphProvider.from_config(
        {
            "edges_name": ("data", "to", "target"),
            "edge_weight_attribute": "gauss_weight",
        },
        graph_data=graph,
    )
    assert isinstance(provider, ProjectionGraphProvider)
    matrix = provider.get_edges().to_dense()
    assert matrix.shape == (2, 3)


def test_from_config_edges_mode_requires_graph_data() -> None:
    with pytest.raises(ValueError, match="graph_data is required"):
        ProjectionGraphProvider.from_config({"edges_name": ("data", "to", "target")})


def test_from_config_ambiguous_raises() -> None:
    with pytest.raises(ValueError, match="at most one of"):
        ProjectionGraphProvider.from_config(
            {"matrix_path": "/fake/path.npz", "edges_name": ("data", "to", "target")},
        )


def test_from_config_invalid_raises() -> None:
    with pytest.raises(ValueError, match="must specify"):
        ProjectionGraphProvider.from_config({"unknown_key": "value"})


def test_projection_graph_provider_row_normalizes_csr_matrix() -> None:
    graph = HeteroData()
    graph["src"].num_nodes = 3
    graph["dst"].num_nodes = 2

    graph[("src", "to", "dst")].edge_index = torch.tensor([[0, 1, 2, 0], [0, 0, 1, 1]])
    graph[("src", "to", "dst")].gauss_weight = torch.tensor([2.0, 8.0, 6.0, 4.0])

    provider = ProjectionGraphProvider(
        graph=graph,
        edges_name=("src", "to", "dst"),
        edge_weight_attribute="gauss_weight",
        row_normalize=True,
    )

    edges = provider.get_edges()
    assert edges.layout == torch.sparse_csr

    expected = torch.tensor([[0.2, 0.8, 0.0], [0.4, 0.0, 0.6]])
    assert torch.allclose(edges.to_dense(), expected, atol=1e-6)


def test_projection_graph_provider_loads_npz_as_csr(tmp_path) -> None:
    file_path = tmp_path / "projection.npz"
    expected = torch.tensor([[0.25, 0.75, 0.0], [0.4, 0.0, 0.6]], dtype=torch.float32)
    save_npz(file_path, csr_matrix(expected.numpy()))

    provider = ProjectionGraphProvider(
        file_path=file_path,
        row_normalize=False,
    )

    edges = provider.get_edges()
    assert edges.layout == torch.sparse_csr
    assert torch.allclose(edges.to_dense(), expected)


class _FixedEdgeBuilder:
    @staticmethod
    def compute_edge_index_from_coords(source_coords: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
        del source_coords, target_coords
        return torch.tensor([[0, 1, 2, 0], [0, 0, 1, 1]])


def _dynamic_provider(attribute) -> DynamicGraphProvider:
    provider = DynamicGraphProvider.__new__(DynamicGraphProvider)
    torch.nn.Module.__init__(provider)
    provider.edge_builder = _FixedEdgeBuilder()
    provider.attributes_config = {"feature": attribute}
    provider._edge_dim = attribute.ndim
    provider._capture_request = None
    provider._captured_graph = None
    return provider


def test_dynamic_graph_provider_routes_source_timedeltas() -> None:
    provider = _dynamic_provider(Timedeltas(node_axis="source", scale_seconds=3600.0))
    src_coords = torch.tensor([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
    dst_coords = torch.tensor([[0.0, 0.0], [0.3, 0.3]])

    edge_attr, edge_index = provider.build_graph(
        src_coords,
        dst_coords,
        src_timedeltas=torch.tensor([-3600.0, 0.0, 7200.0]),
    )

    assert torch.equal(edge_index, torch.tensor([[0, 1, 2, 0], [0, 0, 1, 1]], device=edge_index.device))
    assert torch.equal(
        edge_attr.squeeze(-1),
        torch.tensor([-1.0, 0.0, 2.0, -1.0], device=edge_attr.device),
    )


def test_dynamic_graph_provider_routes_target_timedeltas() -> None:
    provider = _dynamic_provider(Timedeltas(node_axis="target", scale_seconds=3600.0))
    src_coords = torch.tensor([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
    dst_coords = torch.tensor([[0.0, 0.0], [0.3, 0.3]])

    edge_attr, _ = provider.build_graph(
        src_coords,
        dst_coords,
        dst_timedeltas=torch.tensor([1800.0, 3600.0]),
    )

    assert torch.equal(edge_attr.squeeze(-1), torch.tensor([0.5, 0.5, 1.0, 1.0], device=edge_attr.device))


def test_dynamic_graph_provider_batches_timedeltas_with_coordinates() -> None:
    provider = _dynamic_provider(Timedeltas(node_axis="source", scale_seconds=3600.0))
    src_coords = torch.zeros(6, 2)
    dst_coords = torch.zeros(4, 2)

    edge_attr, edge_index = provider.build_graph(
        src_coords,
        dst_coords,
        src_timedeltas=torch.tensor([-3600.0, 0.0, 3600.0, 7200.0, 10800.0, 14400.0]),
        src_batch_sizes=(3, 3),
        dst_batch_sizes=(2, 2),
    )

    assert torch.equal(
        edge_index,
        torch.tensor(
            [[0, 1, 2, 0, 3, 4, 5, 3], [0, 0, 1, 1, 2, 2, 3, 3]],
            device=edge_index.device,
        ),
    )
    assert torch.equal(
        edge_attr.squeeze(-1),
        torch.tensor([-1.0, 0.0, 1.0, -1.0, 2.0, 3.0, 4.0, 2.0], device=edge_attr.device),
    )


def test_dynamic_graph_provider_batches_empty_and_nonempty_samples_on_one_device() -> None:
    provider = _dynamic_provider(Timedeltas(node_axis="source", scale_seconds=3600.0))

    edge_attr, edge_index = provider.build_graph(
        src_coords=torch.zeros(3, 2),
        dst_coords=torch.zeros(4, 2),
        src_timedeltas=torch.tensor([3600.0, 7200.0, 10800.0]),
        src_batch_sizes=(0, 3),
        dst_batch_sizes=(2, 2),
    )

    assert edge_attr.device == edge_index.device
    assert edge_attr.shape == (4, 1)
    assert torch.equal(
        edge_attr.squeeze(-1),
        torch.tensor([1.0, 2.0, 3.0, 1.0], device=edge_attr.device),
    )


def test_dynamic_graph_capture_keeps_timedeltas_and_encoded_edges() -> None:
    provider = _dynamic_provider(Timedeltas(node_axis="source", scale_seconds=3600.0))
    provider.capture_next_graph("obs", "hidden")
    src_coords = torch.tensor([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
    dst_coords = torch.tensor([[0.0, 0.0], [0.3, 0.3]])
    src_timedeltas = torch.tensor([-3600.0, 0.0, 7200.0])

    provider.get_edges(
        src_coords=src_coords,
        dst_coords=dst_coords,
        src_timedeltas=src_timedeltas,
        shard_edges=False,
        act_checkpoint=False,
    )
    graph = provider.consume_captured_graph()

    assert graph is not None
    assert torch.equal(graph["obs"].timedeltas, src_timedeltas)
    assert torch.equal(
        graph[("obs", "to", "hidden")].feature.squeeze(-1),
        torch.tensor([-1.0, 0.0, 2.0, -1.0]),
    )


def test_dynamic_graph_provider_checkpoints_timedelta_inputs() -> None:
    provider = _dynamic_provider(Timedeltas(node_axis="source", scale_seconds=3600.0))
    device = provider.attributes_config["feature"].device

    edge_attr, edge_index, shard_sizes = provider.get_edges(
        src_coords=torch.tensor([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]], device=device),
        dst_coords=torch.tensor([[0.0, 0.0], [0.3, 0.3]], device=device),
        src_timedeltas=torch.tensor([-3600.0, 0.0, 7200.0], device=device),
        shard_edges=False,
    )

    assert edge_attr.shape == (4, 1)
    assert edge_index.shape == (2, 4)
    assert shard_sizes is None


def test_dynamic_graph_provider_keeps_coordinate_attributes_working() -> None:
    provider = _dynamic_provider(EdgeLength(norm=None))
    src_coords = torch.tensor([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
    dst_coords = torch.tensor([[0.0, 0.0], [0.3, 0.3]])

    edge_attr, edge_index = provider.build_graph(src_coords, dst_coords)
    source_nodes = NodeStorage()
    source_nodes.x = src_coords
    target_nodes = NodeStorage()
    target_nodes.x = dst_coords
    expected = EdgeLength(norm=None)(
        x=(source_nodes, target_nodes),
        edge_index=edge_index,
    )

    assert torch.allclose(edge_attr, expected)
