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

from anemoi.models.distributed.khop_edges import build_graph_partition
from anemoi.models.layers import graph_provider as graph_provider_module
from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.models.layers.graph_provider import StaticGraphProvider


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


def test_projection_graph_provider_does_not_mutate_graph_weights() -> None:
    graph = _make_graph_with_edges()
    graph["data"].area = torch.tensor([2.0, 3.0, 4.0])
    original_weights = graph["data", "to", "target"].gauss_weight.clone()

    ProjectionGraphProvider(
        graph=graph,
        edges_name=("data", "to", "target"),
        edge_weight_attribute="gauss_weight",
        src_node_weight_attribute="area",
    )

    torch.testing.assert_close(graph["data", "to", "target"].gauss_weight, original_weights)


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


def test_projection_graph_provider_retains_requested_dtype() -> None:
    graph = HeteroData()
    graph["src"].num_nodes = 3
    graph["dst"].num_nodes = 2
    graph[("src", "to", "dst")].edge_index = torch.tensor([[0, 1, 2, 0], [0, 0, 1, 1]])

    provider = ProjectionGraphProvider(
        graph=graph,
        edges_name=("src", "to", "dst"),
    )

    float64_matrix = provider.get_edges(dtype=torch.float64)

    assert float64_matrix.dtype == torch.float64
    assert provider.get_edges(dtype=torch.float64) is float64_matrix


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


@pytest.fixture
def cpu_default_device():
    """Pin CPU as default device; other tests in this directory leak a CUDA default."""
    previous = torch.get_default_device()
    torch.set_default_device("cpu")
    yield
    torch.set_default_device(previous)


def _static_provider(trainable_size: int = 0) -> StaticGraphProvider:
    edge_index = torch.tensor([[0, 1, 2, 3, 1, 2], [0, 0, 1, 2, 3, 3]])
    graph = HeteroData()
    graph.edge_index = edge_index
    graph.edge_length = torch.arange(edge_index.shape[1], dtype=torch.float32).unsqueeze(-1)
    return StaticGraphProvider(
        graph=graph,
        edge_attributes=["edge_length"],
        src_size=4,
        dst_size=4,
        trainable_size=trainable_size,
    )


@pytest.mark.usefixtures("cpu_default_device")
def test_static_graph_provider_sharded_edges_match_unsharded_when_not_distributed() -> None:
    provider = _static_provider()
    attr_sharded, index_sharded, sizes = provider.get_edges(batch_size=2)
    attr_full, index_full, _ = provider.get_edges(batch_size=2, shard_edges=False)
    assert sizes is None
    torch.testing.assert_close(index_sharded, index_full)
    torch.testing.assert_close(attr_sharded, attr_full)


@pytest.mark.usefixtures("cpu_default_device")
def test_static_graph_provider_caches_edge_sharding(monkeypatch: pytest.MonkeyPatch) -> None:
    """The edge partition is computed once per layout; later calls reuse it without host syncs."""
    provider = _static_provider(trainable_size=2)
    rank, comm_size = 1, 2

    class _FakeGroup:
        def size(self) -> int:
            return comm_size

    def _fake_shard(tensor: torch.Tensor, dim: int, sizes: list[int] | None, _group: object) -> torch.Tensor:
        return torch.split(tensor, sizes, dim=dim)[rank]

    partition_calls = []

    def _counting_partition(*args: object, **kwargs: object) -> object:
        partition_calls.append(1)
        return build_graph_partition(*args, **kwargs)

    monkeypatch.setattr(graph_provider_module, "model_is_distributed", lambda group: group is not None)
    monkeypatch.setattr(graph_provider_module, "shard_tensor", _fake_shard)
    monkeypatch.setattr(graph_provider_module, "build_graph_partition", _counting_partition)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: rank)  # noqa: ARG005

    group = _FakeGroup()
    first = provider.get_edges(batch_size=1, model_comm_group=group)
    second = provider.get_edges(batch_size=1, model_comm_group=group)
    assert len(partition_calls) == 1

    # Reference: the uncached computation for rank 1 of 2.
    attr_full, index_full, _ = provider.get_edges(batch_size=1, shard_edges=False)
    expected_sizes = build_graph_partition(index_full, comm_size, (4, 4)).edge_splits
    for attr, index, sizes in (first, second):
        assert sizes == expected_sizes
        torch.testing.assert_close(index, torch.split(index_full, expected_sizes, dim=1)[rank])
        torch.testing.assert_close(attr, torch.split(attr_full, expected_sizes, dim=0)[rank])

    # Trainable edge features are still recomputed per call, so gradients reach them.
    second[0].sum().backward()
    assert provider.trainable.trainable.grad is not None
