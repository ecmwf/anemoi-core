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

from anemoi.graphs.generate.icosahedral_diamonds import IcosahedralDiamonds
from anemoi.graphs.nodes import TriNodes
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder


@pytest.mark.parametrize("resolution", [0, 2])
def test_init(resolution: list[int]):
    """Test TrirefinedIcosahedralNodes initialization."""

    node_builder = TriNodes(resolution, "test_nodes")
    assert isinstance(node_builder, BaseNodeBuilder)
    assert isinstance(node_builder, TriNodes)


def test_get_coordinates():
    """Test get_coordinates method."""
    node_builder = TriNodes(2, "test_nodes")
    coords = node_builder.get_coordinates()
    assert isinstance(coords, torch.Tensor)
    assert coords.shape == (162, 2)


def test_update_graph():
    """Test update_graph method."""
    node_builder = TriNodes(1, "test_nodes")
    graph = HeteroData()
    graph = node_builder.update_graph(graph, {})
    assert "_resolutions" in graph["test_nodes"]
    assert "_nx_graph" in graph["test_nodes"]
    assert "_node_ordering" in graph["test_nodes"]
    assert len(graph["test_nodes"]["_node_ordering"]) == graph["test_nodes"].num_nodes


def test_diamond_ordering():
    """With ordering "diamonds", nodes neighbouring on the diamond lattice are neighbours on the sphere."""
    node_builder = TriNodes(3, "test_nodes", ordering="diamonds")
    coords = node_builder.get_coordinates().numpy().astype(np.float64)
    lat, lon = coords[:, 0], coords[:, 1]
    xyz = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)

    edges = np.concatenate([IcosahedralDiamonds(3).triangles()[:, pair] for pair in ([0, 1], [1, 2], [2, 0])])
    lengths = np.arccos(np.clip(np.einsum("ij,ij->i", xyz[edges[:, 0]], xyz[edges[:, 1]]), -1, 1))
    assert lengths.max() < 1.5 * lengths.min()
    assert np.allclose(xyz[0], -xyz[-1], atol=1e-6)


def test_unknown_ordering():
    with pytest.raises(ValueError, match="Unknown node ordering"):
        TriNodes(2, "test_nodes", ordering="random")
