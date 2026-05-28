# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.models.layers.graph_provider import FileGraphProvider
from anemoi.models.layers.graph_provider import ProjectionGraphProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_SRC_NODES = 5
NUM_DST_NODES = 4
NUM_EDGES = 8
EDGE_ATTR_DIM = 3


def _make_fake_graph(seed: int = 0) -> HeteroData:
    """Create a small fake HeteroData graph with random edges and attributes."""
    rng = torch.Generator().manual_seed(seed)

    graph = HeteroData()
    graph.edge_index = torch.stack(
        [
            torch.randint(0, NUM_SRC_NODES, (NUM_EDGES,), generator=rng),
            torch.randint(0, NUM_DST_NODES, (NUM_EDGES,), generator=rng),
        ]
    )
    graph["edge_length"] = torch.randn(NUM_EDGES, EDGE_ATTR_DIM, generator=rng)
    graph.edge_attribute_names = ["edge_length"]
    graph.src_size = NUM_SRC_NODES
    graph.dst_size = NUM_DST_NODES
    return graph


@pytest.fixture()
def graph_dir(tmp_path: Path) -> Path:
    """Save multiple fake graphs to a temporary directory and return the path."""
    graphs_path = tmp_path / "graphs"
    graphs_path.mkdir()

    for i in range(4):
        g = _make_fake_graph(seed=i)
        torch.save(g, graphs_path / f"graph_{i:03d}.pt")

    return graphs_path


# ---------------------------------------------------------------------------
# FileGraphProvider tests
# ---------------------------------------------------------------------------


def test_file_graph_provider_len(graph_dir: Path) -> None:
    """Provider reports correct number of graph files."""
    provider = FileGraphProvider(
        graph_dir=graph_dir,
        num_workers=1,
        pin_memory=False,
    )
    assert len(provider) == 4


def test_file_graph_provider_edge_dim(graph_dir: Path) -> None:
    """edge_dim matches the attribute width."""
    provider = FileGraphProvider(
        graph_dir=graph_dir,
        num_workers=1,
        pin_memory=False,
    )
    assert provider.edge_dim == EDGE_ATTR_DIM


def test_file_graph_provider_iteration(graph_dir: Path) -> None:
    """Iterating over the provider yields all graphs."""
    provider = FileGraphProvider(
        graph_dir=graph_dir,
        num_workers=1,
        pin_memory=False,
    )

    loaded = list(provider)
    assert len(loaded) == 4

    for g in loaded:
        assert hasattr(g, "edge_index")
        assert g.edge_index.shape == (2, NUM_EDGES)


def test_file_graph_provider_get_edges_no_shard(graph_dir: Path) -> None:
    """get_edges returns correct shapes without sharding."""
    provider = FileGraphProvider(
        graph_dir=graph_dir,
        num_workers=1,
        pin_memory=False,
    )

    # Load a graph and pass it to get_edges
    graph = provider._dataset[0]
    edge_attr, edge_index, shard_sizes = provider.get_edges(batch_size=1, graph=graph, shard_edges=False)

    assert edge_attr.shape == (NUM_EDGES, EDGE_ATTR_DIM)
    assert edge_index.shape == (2, NUM_EDGES)
    assert shard_sizes is None


def test_file_graph_provider_get_edges_batch_expansion(graph_dir: Path) -> None:
    """get_edges expands edges correctly for batch_size > 1."""
    provider = FileGraphProvider(
        graph_dir=graph_dir,
        num_workers=1,
        pin_memory=False,
    )

    batch_size = 3
    graph = provider._dataset[0]
    edge_attr, edge_index, _ = provider.get_edges(batch_size=batch_size, graph=graph, shard_edges=False)

    assert edge_attr.shape == (NUM_EDGES * batch_size, EDGE_ATTR_DIM)
    assert edge_index.shape == (2, NUM_EDGES * batch_size)


def test_file_graph_provider_get_edges_default_graph(graph_dir: Path) -> None:
    """get_edges falls back to first graph when graph=None."""
    provider = FileGraphProvider(
        graph_dir=graph_dir,
        num_workers=1,
        pin_memory=False,
    )

    edge_attr, edge_index, _ = provider.get_edges(batch_size=1, shard_edges=False)
    assert edge_attr.shape[0] == NUM_EDGES


def test_file_graph_provider_missing_dir(tmp_path: Path) -> None:
    """Raises FileNotFoundError for a nonexistent directory."""
    with pytest.raises(FileNotFoundError):
        FileGraphProvider(
            graph_dir=tmp_path / "nonexistent",
            num_workers=1,
            pin_memory=False,
        )


def test_file_graph_provider_empty_dir(tmp_path: Path) -> None:
    """Raises RuntimeError when directory has no graph files."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError):
        FileGraphProvider(
            graph_dir=empty,
            num_workers=1,
            pin_memory=False,
        )


# ---------------------------------------------------------------------------
# ProjectionGraphProvider tests
# ---------------------------------------------------------------------------


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

    matrix = provider.get_edges().to_dense()
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

    matrix = provider.get_edges().to_dense()
    assert matrix.shape == (graph["dst"].num_nodes, graph["src"].num_nodes)
    row_sums = matrix.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
