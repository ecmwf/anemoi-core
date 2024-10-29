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
import zarr
from torch_geometric.data import HeteroData

from anemoi.graphs.nodes.attributes import AreaWeights
from anemoi.graphs.nodes.attributes import UniformWeights
from anemoi.graphs.nodes.builders import from_file


def test_init(mocker, mock_zarr_dataset):
    """Test ZarrDatasetNodes initialization."""
    mocker.patch.object(from_file, "open_dataset", return_value=mock_zarr_dataset)
    node_builder = from_file.ZarrDatasetNodes("dataset.zarr", name="test_nodes")

    assert isinstance(node_builder, from_file.BaseNodeBuilder)
    assert isinstance(node_builder, from_file.ZarrDatasetNodes)


def test_fail():
    """Test ZarrDatasetNodes with invalid dataset."""
    node_builder = from_file.ZarrDatasetNodes("invalid_path.zarr", name="test_nodes")
    with pytest.raises(zarr.errors.PathNotFoundError):
        node_builder.update_graph(HeteroData())


def test_register_nodes(mocker, mock_zarr_dataset):
    """Test ZarrDatasetNodes register correctly the nodes."""
    mocker.patch.object(from_file, "open_dataset", return_value=mock_zarr_dataset)
    node_builder = from_file.ZarrDatasetNodes("dataset.zarr", name="test_nodes")
    graph = HeteroData()

    graph = node_builder.register_nodes(graph)

    assert graph["test_nodes"].x is not None
    assert isinstance(graph["test_nodes"].x, torch.Tensor)
    assert graph["test_nodes"].x.shape == (mock_zarr_dataset.num_nodes, 2)
    assert graph["test_nodes"].node_type == "ZarrDatasetNodes"


@pytest.mark.parametrize("attr_class", [UniformWeights, AreaWeights])
def test_register_attributes(mocker, graph_with_nodes: HeteroData, attr_class):
    """Test ZarrDatasetNodes register correctly the weights."""
    mocker.patch.object(from_file, "open_dataset", return_value=None)
    node_builder = from_file.ZarrDatasetNodes("dataset.zarr", name="test_nodes")
    config = {"test_attr": {"_target_": f"anemoi.graphs.nodes.attributes.{attr_class.__name__}"}}

    graph = node_builder.register_attributes(graph_with_nodes, config)

    assert graph["test_nodes"]["test_attr"] is not None
    assert isinstance(graph["test_nodes"]["test_attr"], torch.Tensor)
    assert graph["test_nodes"]["test_attr"].shape[0] == graph["test_nodes"].x.shape[0]
