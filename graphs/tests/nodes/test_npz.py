# (C) Copyright 2024 Anemoi contributors.
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

from anemoi.graphs.nodes.attributes import SphericalAreaWeights
from anemoi.graphs.nodes.attributes import UniformWeights
from anemoi.graphs.nodes.builders.from_file import NPZFileNodes


@pytest.mark.parametrize("resolution", ["o16", "o48", "5km5"])
def test_init(mock_grids_path: tuple[str, int], resolution: str):
    """Test NPZFileNodes initialization."""
    grid_definition_path, _ = mock_grids_path
    node_builder = NPZFileNodes(
        npz_file=grid_definition_path + f"/grid-{resolution}.npz",
        name="test_nodes",
        lat_key="latitudes",
        lon_key="longitudes",
    )
    assert isinstance(node_builder, NPZFileNodes)


def test_fail_init_wrong_path():
    """Test NPZFileNodes initialization with invalid path."""
    node_builder = NPZFileNodes("invalid_path.myext", "test_nodes")
    with pytest.raises(AssertionError):
        node_builder.get_coordinates()


@pytest.mark.parametrize("resolution", ["o16", "o48", "5km5"])
def test_register_nodes(mock_grids_path: str, resolution: str):
    """Test NPZFileNodes register correctly the nodes."""
    graph = HeteroData()
    grid_definition_path, num_nodes = mock_grids_path
    node_builder = NPZFileNodes(npz_file=grid_definition_path + f"/grid-{resolution}.npz", name="test_nodes")

    graph = node_builder.register_nodes(graph)

    assert graph["test_nodes"].x is not None
    assert isinstance(graph["test_nodes"].x, torch.Tensor)
    assert graph["test_nodes"].x.shape == (num_nodes, 2)
    assert graph["test_nodes"].node_type == "NPZFileNodes"


@pytest.mark.parametrize("attr_class", [UniformWeights, SphericalAreaWeights])
@pytest.mark.parametrize("resolution", ["o16", "o48", "5km5"])
def test_register_attributes(
    graph_with_nodes: HeteroData, mock_grids_path: tuple[str, int], attr_class, resolution: str
):
    """Test NPZFileNodes register correctly the weights."""
    grid_definition_path, _ = mock_grids_path
    node_builder = NPZFileNodes(npz_file=grid_definition_path + f"/grid-{resolution}.npz", name="test_nodes")
    config = {"test_attr": {"_target_": f"anemoi.graphs.nodes.attributes.{attr_class.__name__}"}}

    graph = node_builder.register_attributes(graph_with_nodes, config)

    assert graph["test_nodes"]["test_attr"] is not None
    assert isinstance(graph["test_nodes"]["test_attr"], torch.Tensor)
    assert graph["test_nodes"]["test_attr"].shape[0] == graph["test_nodes"].x.shape[0]
