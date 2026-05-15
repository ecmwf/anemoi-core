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
from anemoi.graphs.nodes.attributes import GridsMask
from anemoi.graphs.nodes.attributes.masks import BaseCombineAnemoiDatasetsMask


def test_cutout_mask(mocker, graph_with_nodes: HeteroData, mock_anemoi_dataset_cutout):
    """Test attribute builder for CutOutMask."""
    # Add dataset attribute required by CutOutMask
    graph_with_nodes["test_nodes"]["_dataset"] = {}

    mocker.patch("anemoi.datasets.open_dataset", return_value=mock_anemoi_dataset_cutout)
    mask = CutOutMask().compute(graph_with_nodes, "test_nodes")

    assert mask is not None
    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.bool
    assert mask.shape[0] == graph_with_nodes["test_nodes"].x.shape[0]


def test_get_mask_from_grid_size():
    grid_sizes1 = (1, 2, 3, 2, 1)
    grid_sizes2 = (4,)
    grid_ids1 = [0, 1, 4]
    grid_ids2 = [1]

    grids_mask1 = BaseCombineAnemoiDatasetsMask.get_mask_from_grid_sizes(grid_sizes1, grid_ids1)
    grids_mask2 = BaseCombineAnemoiDatasetsMask.get_mask_from_grid_sizes(grid_sizes1, grid_ids2)

    with pytest.raises(AssertionError):
        BaseCombineAnemoiDatasetsMask.get_mask_from_grid_sizes(grid_sizes2, grid_ids2)

    assert all(grids_mask1 == torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 1], dtype=torch.bool))
    assert all(grids_mask2 == torch.tensor([0, 1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool))


@pytest.mark.parametrize("mask_class", [CutOutMask, GridsMask])
def test_combined_datasets_mask_missing_dataset(graph_with_nodes: HeteroData, mask_class):
    """Test CutOutMask fails when dataset attribute is missing."""
    node_attr_builder = mask_class()
    with pytest.raises(AssertionError):
        node_attr_builder.compute(graph_with_nodes, "test_nodes")


@pytest.mark.parametrize("grids_0, grids_1", [([0], [1])])
def test_cutout_mask_grids(mocker, graph_with_nodes: HeteroData, mock_anemoi_dataset_cutout, grids_0, grids_1):
    """Test CutOutMask default, [0], and [1] grid arguments."""
    graph_with_nodes["test_nodes"]["_dataset"] = {}
    mocker.patch("anemoi.datasets.open_dataset", return_value=mock_anemoi_dataset_cutout)

    mask_default = CutOutMask()
    mask_0 = CutOutMask(grids=grids_0)
    mask_1 = CutOutMask(grids=grids_1)
    mask_default = mask_default.compute(graph_with_nodes, "test_nodes")
    mask_0 = mask_0.compute(graph_with_nodes, "test_nodes")
    mask_1 = mask_1.compute(graph_with_nodes, "test_nodes")

    # Default and explicit [0] should be equal
    assert torch.equal(mask_default, mask_0)

    # [0] and [1] should be complementary in mock_anemoi_dataset_cutout(non-overlapping and covering all nodes)
    assert isinstance(mask_1, torch.Tensor)
    assert mask_1.dtype == torch.bool
    assert mask_1.shape == mask_0.shape
    assert not torch.any(mask_0 & mask_1), "Grids 0 and 1 should not overlap."
    assert torch.all(mask_0 | mask_1), "Grids 0 and 1 together should cover all nodes."


if __name__ == "__main__":
    pytest.main([__file__])
