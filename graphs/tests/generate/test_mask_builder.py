# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import HeteroData

from anemoi.graphs.generate.masks import KNNAreaMaskBuilder


def test_init():
    """Test KNNAreaMaskBuilder initialization."""
    mask_builder1 = KNNAreaMaskBuilder("nodes")
    mask_builder2 = KNNAreaMaskBuilder("nodes", margin_radius_km=120)
    mask_builder3 = KNNAreaMaskBuilder("nodes", mask_attr_name="mask")
    mask_builder4 = KNNAreaMaskBuilder("nodes", margin_radius_km=120, mask_attr_name="mask")

    assert isinstance(mask_builder1, KNNAreaMaskBuilder)
    assert isinstance(mask_builder2, KNNAreaMaskBuilder)
    assert isinstance(mask_builder3, KNNAreaMaskBuilder)
    assert isinstance(mask_builder4, KNNAreaMaskBuilder)

    assert isinstance(mask_builder1.nearest_neighbour, NearestNeighbors)
    assert isinstance(mask_builder2.nearest_neighbour, NearestNeighbors)
    assert isinstance(mask_builder3.nearest_neighbour, NearestNeighbors)
    assert isinstance(mask_builder4.nearest_neighbour, NearestNeighbors)


@pytest.mark.parametrize("margin", [-1, "120", None])
def test_fail_init_wrong_margin(margin: int):
    """Test KNNAreaMaskBuilder initialization with invalid margin."""
    with pytest.raises(AssertionError):
        KNNAreaMaskBuilder("nodes", margin_radius_km=margin)


@pytest.mark.parametrize("mask", [None, "mask"])
def test_fit(graph_with_nodes: HeteroData, mask: str):
    """Test KNNAreaMaskBuilder fit."""
    mask_builder = KNNAreaMaskBuilder("test_nodes", mask_attr_name=mask)
    assert not hasattr(mask_builder.nearest_neighbour, "n_samples_fit_")

    mask_builder.fit(graph_with_nodes)

    assert mask_builder.nearest_neighbour.n_samples_fit_ == graph_with_nodes["test_nodes"].num_nodes


def test_fit_fail(graph_with_nodes):
    """Test KNNAreaMaskBuilder fit with wrong graph."""
    mask_builder = KNNAreaMaskBuilder("wrong_nodes")
    with pytest.raises(AssertionError):
        mask_builder.fit(graph_with_nodes)
