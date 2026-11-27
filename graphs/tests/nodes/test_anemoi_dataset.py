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
from omegaconf import OmegaConf
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


@pytest.mark.parametrize("as_config", [True, False], ids=["dictconfig", "dict"])
def test_unset_dataset_options_are_not_passed_on(mocker, mock_anemoi_dataset, as_config):
    """Options set to None are dropped, as the training readers do: ``open_dataset`` treats them as given."""
    open_dataset = mocker.patch("anemoi.datasets.open_dataset", return_value=mock_anemoi_dataset)
    options = {"dataset": "dataset.zarr", "frequency": None, "select": None, "step_frequency": 1}

    node_builder = from_file.AnemoiDatasetNodes(OmegaConf.create(options) if as_config else options, name="test_nodes")
    node_builder.register_nodes(HeteroData())

    open_dataset.assert_called_once_with({"dataset": "dataset.zarr", "step_frequency": 1})


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
