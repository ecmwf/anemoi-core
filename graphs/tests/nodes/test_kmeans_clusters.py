# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.nodes import KMeansClusteredNodes


@pytest.fixture
def graph_with_reference_nodes() -> HeteroData:
    """Graph with 200 reference nodes over the globe, half of them masked in."""
    rng = np.random.default_rng(42)
    num_nodes = 200
    coords = np.stack([np.arcsin(rng.uniform(-1, 1, num_nodes)), rng.uniform(-np.pi, np.pi, num_nodes)], axis=1)
    graph = HeteroData()
    graph["reference_nodes"].x = torch.tensor(coords, dtype=torch.float32)
    graph["reference_nodes"].mask = torch.tensor([True] * 100 + [False] * 100).unsqueeze(-1)
    return graph


def test_init():
    """Test KMeansClusteredNodes initialization."""
    KMeansClusteredNodes("reference_nodes", ratio=4, name="test_nodes")


@pytest.mark.parametrize("ratio", [-1, 0, 1])
def test_fail_init(ratio: float):
    """Test KMeansClusteredNodes initialization with an invalid ratio."""
    with pytest.raises(ValueError, match="ratio must be > 1"):
        KMeansClusteredNodes("reference_nodes", ratio=ratio, name="test_nodes")


def test_register_nodes(graph_with_reference_nodes: HeteroData):
    """Test KMeansClusteredNodes register correctly the nodes."""
    builder = KMeansClusteredNodes(
        "reference_nodes", ratio=4, name="test_nodes", kmeans_kwargs={"random_state": 42, "n_init": 3}
    )

    graph = builder.update_graph(graph_with_reference_nodes)

    assert graph["test_nodes"].num_nodes == 50
    assert graph["test_nodes"].x.shape == (50, 2)
    latitudes, longitudes = graph["test_nodes"].x.T
    assert latitudes.abs().max() <= np.pi / 2
    assert longitudes.abs().max() <= np.pi


def test_register_nodes_with_mask(graph_with_reference_nodes: HeteroData):
    """Test KMeansClusteredNodes only clusters the reference nodes inside the mask."""
    builder = KMeansClusteredNodes(
        "reference_nodes",
        ratio=4,
        name="test_nodes",
        mask_attr_name="mask",
        kmeans_kwargs={"random_state": 42, "n_init": 3},
    )

    graph = builder.update_graph(graph_with_reference_nodes)

    assert graph["test_nodes"].num_nodes == 25


def test_register_nodes_with_weights(graph_with_reference_nodes: HeteroData):
    """Test KMeansClusteredNodes with weighted reference nodes."""
    latitudes = graph_with_reference_nodes["reference_nodes"].x[:, 0]
    graph_with_reference_nodes["reference_nodes"]["area_weight"] = latitudes.cos().unsqueeze(-1)
    builder = KMeansClusteredNodes(
        "reference_nodes",
        ratio=4,
        name="test_nodes",
        node_weight_attribute_name="area_weight",
        kmeans_kwargs={"random_state": 42, "n_init": 3},
    )

    graph = builder.update_graph(graph_with_reference_nodes)

    assert graph["test_nodes"].num_nodes == 50


def test_fail_missing_reference_nodes(graph_with_reference_nodes: HeteroData):
    """Test KMeansClusteredNodes fails if the reference nodes are not in the graph."""
    builder = KMeansClusteredNodes("missing_nodes", ratio=4, name="test_nodes")

    with pytest.raises(AssertionError, match="not found in the graph"):
        builder.update_graph(graph_with_reference_nodes)


def test_fail_missing_mask_attribute(graph_with_reference_nodes: HeteroData):
    """Test KMeansClusteredNodes fails if the mask attribute is not in the reference nodes."""
    builder = KMeansClusteredNodes("reference_nodes", ratio=4, name="test_nodes", mask_attr_name="missing_mask")

    with pytest.raises(AssertionError, match="not found in the reference nodes"):
        builder.update_graph(graph_with_reference_nodes)
