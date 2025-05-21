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

from anemoi.graphs.nodes.attributes import CutOutMask


def test_cutout_mask(mocker, graph_with_nodes: HeteroData, mock_anemoi_dataset_cutout):
    """Test attribute builder for CutOutMask."""
    # Add dataset attribute required by CutOutMask
    graph_with_nodes["test_nodes"]["_dataset"] = {"cutout": None}

    mocker.patch(
        "anemoi.graphs.nodes.attributes.masks.open_dataset",
        return_value=mock_anemoi_dataset_cutout,
    )
    mask = CutOutMask().compute(graph_with_nodes, "test_nodes")

    assert mask is not None
    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.bool
    assert mask.shape[0] == graph_with_nodes["test_nodes"].x.shape[0]


def test_cutout_mask_missing_dataset(graph_with_nodes: HeteroData):
    """Test CutOutMask fails when dataset attribute is missing."""
    node_attr_builder = CutOutMask()
    with pytest.raises(AssertionError):
        node_attr_builder.compute(graph_with_nodes, "test_nodes")


def test_cutout_mask_missing_cutout(graph_with_nodes: HeteroData):
    """Test CutOutMask fails when cutout key is missing."""
    graph_with_nodes["test_nodes"]["_dataset"] = {}

    node_attr_builder = CutOutMask()
    with pytest.raises(AssertionError):
        node_attr_builder.compute(graph_with_nodes, "test_nodes")
