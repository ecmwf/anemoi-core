# (C) Copyright 2024- Anemoi contributors.
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

from anemoi.graphs.nodes.attributes import CutOutMask, GridsMask


@pytest.mark.parametrize("mask_class", [CutOutMask, GridsMask])
def test_combined_datasets_mask(mocker, graph_with_nodes: HeteroData, mock_zarr_dataset_cutout, mask_class):
    """Test attribute builder for CutOutMask."""
    # Add dataset attribute required by CutOutMask
    graph_with_nodes["test_nodes"]["_dataset"] = {}

    mocker.patch("anemoi.graphs.nodes.attributes.masks.open_dataset", return_value=mock_zarr_dataset_cutout)
    mask = mask_class().compute(graph_with_nodes, "test_nodes")

    assert mask is not None
    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.bool
    assert mask.shape[0] == graph_with_nodes["test_nodes"].x.shape[0]


@pytest.mark.parametrize("mask_class", [CutOutMask, GridsMask])
def test_combined_datasets_mask_missing_dataset(graph_with_nodes: HeteroData, mask_class):
    """Test CutOutMask fails when dataset attribute is missing."""
    node_attr_builder = mask_class()
    with pytest.raises(AssertionError):
        node_attr_builder.compute(graph_with_nodes, "test_nodes")


@pytest.mark.parametrize("mask_class", [CutOutMask, GridsMask])
def test_combined_datasets_mask_missing_grids(mocker, graph_with_nodes: HeteroData, mock_zarr_dataset, mask_class):
    """Test CutOutMask fails when grids is not a dataset property."""
    graph_with_nodes["test_nodes"]["_dataset"] = {}
    mocker.patch("anemoi.graphs.nodes.attributes.masks.open_dataset", return_value=mock_zarr_dataset)

    node_attr_builder = mask_class()
    with pytest.raises(AssertionError):
        node_attr_builder.compute(graph_with_nodes, "test_nodes")
