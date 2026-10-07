# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.nodes.attributes import CutOutMask
from anemoi.graphs.nodes.attributes import SphericalAreaWeights
from anemoi.graphs.nodes.attributes import UniformWeights
from anemoi.graphs.nodes.builders.from_vectors import LatLonNodes

lats = [45.0, 45.0, 40.0, 40.0]
lons = [5.0, 10.0, 10.0, 5.0]


def test_init():
    """Test LatLonNodes initialization."""
    node_builder = LatLonNodes(latitudes=lats, longitudes=lons, name="test_nodes")
    assert isinstance(node_builder, LatLonNodes)


def test_fail_init_length_mismatch():
    """Test LatLonNodes initialization with invalid argument."""
    lons = [5.0, 10.0, 10.0, 5.0, 5.0]

    with pytest.raises(AssertionError):
        LatLonNodes(latitudes=lats, longitudes=lons, name="test_nodes")


def test_fail_init_missing_argument():
    """Test NPZFileNodes initialization with missing argument."""
    with pytest.raises(TypeError):
        LatLonNodes(name="test_nodes")


def test_register_nodes():
    """Test LatLonNodes register correctly the nodes."""
    graph = HeteroData()
    node_builder = LatLonNodes(latitudes=lats, longitudes=lons, name="test_nodes")
    node_builder.register_nodes(graph)

    assert graph["test_nodes"].x is not None
    assert isinstance(graph["test_nodes"].x, torch.Tensor)
    assert graph["test_nodes"].x.shape == (len(lats), 2)
    assert graph["test_nodes"].node_type == "LatLonNodes"


@pytest.mark.parametrize("attr_class", [UniformWeights, SphericalAreaWeights])
def test_register_attributes(graph_with_nodes: HeteroData, attr_class):
    """Test LatLonNodes register correctly the weights."""
    node_builder = LatLonNodes(latitudes=lats, longitudes=lons, name="test_nodes")

    attr = attr_class(name="test_attr")
    node_builder.register_attributes(graph_with_nodes, [attr])

    assert graph_with_nodes["test_nodes"]["test_attr"] is not None
    assert isinstance(graph_with_nodes["test_nodes"]["test_attr"], torch.Tensor)
    assert graph_with_nodes["test_nodes"]["test_attr"].shape[0] == graph_with_nodes["test_nodes"].x.shape[0]


def test_register_attributes_default_name(graph_with_nodes: HeteroData):
    """Test the class-level name is used when no name is given."""
    node_builder = LatLonNodes(latitudes=lats, longitudes=lons, name="test_nodes")
    node_builder.register_attributes(graph_with_nodes, [UniformWeights()])

    assert "area_weights" in graph_with_nodes["test_nodes"]


def test_register_attributes_fail_without_name(graph_with_nodes: HeteroData):
    """Test registering an attribute without name raises an error."""
    node_builder = LatLonNodes(latitudes=lats, longitudes=lons, name="test_nodes")

    with pytest.raises(ValueError):
        node_builder.register_attributes(graph_with_nodes, [CutOutMask()])
