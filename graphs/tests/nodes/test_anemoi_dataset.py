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
from anemoi.graphs.nodes.builders import from_file


def test_init(mocker, mock_anemoi_dataset):
    """Test AnemoiDatasetNodes initialization."""
    mocker.patch("anemoi.datasets.open_dataset", return_value=mock_anemoi_dataset)

    node_builder = from_file.AnemoiDatasetNodes("anemoi-dataset", name="test_nodes")

    assert isinstance(node_builder, from_file.BaseNodeBuilder)
    assert isinstance(node_builder, from_file.AnemoiDatasetNodes)


def test_fail():
    """Test AnemoiDatasetNodes with invalid dataset."""
    node_builder = from_file.AnemoiDatasetNodes("invalid_path.zarr", name="test_nodes")
    with pytest.raises(Exception):
        node_builder.update_graph(HeteroData())


def test_register_nodes(mocker, mock_anemoi_dataset):
    """Test AnemoiDatasetNodes register correctly the nodes."""
    mocker.patch("anemoi.datasets.open_dataset", return_value=mock_anemoi_dataset)

    node_builder = from_file.AnemoiDatasetNodes("dataset.zarr", name="test_nodes")
    graph = HeteroData()

    graph = node_builder.register_nodes(graph)

    assert graph["test_nodes"].x is not None
    assert isinstance(graph["test_nodes"].x, torch.Tensor)
    assert graph["test_nodes"].x.shape == (mock_anemoi_dataset.num_nodes, 2)
    assert graph["test_nodes"].node_type == "AnemoiDatasetNodes"


@pytest.mark.parametrize("attr_class", [UniformWeights, SphericalAreaWeights])
def test_register_attributes(mocker, graph_with_nodes: HeteroData, attr_class):
    """Test AnemoiDatasetNodes register correctly the weights."""
    mocker.patch("anemoi.datasets.open_dataset", return_value=None)
    node_builder = from_file.AnemoiDatasetNodes("dataset.zarr", name="test_nodes")
    config = {"test_attr": {"_target_": f"anemoi.graphs.nodes.attributes.{attr_class.__name__}"}}

    graph = node_builder.register_attributes(graph_with_nodes, config)

    assert graph["test_nodes"]["test_attr"] is not None
    assert isinstance(graph["test_nodes"]["test_attr"], torch.Tensor)
    assert graph["test_nodes"]["test_attr"].shape[0] == graph["test_nodes"].x.shape[0]
