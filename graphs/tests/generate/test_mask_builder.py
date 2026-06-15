# (C) Copyright 2024 Anemoi contributors.
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
from scipy.spatial import cKDTree
from torch_geometric.data import HeteroData

from anemoi.graphs import EARTH_RADIUS
from anemoi.graphs.generate.masks import PYG_LIB_AVAILABLE
from anemoi.graphs.generate.masks import AreaMaskBuilder


def test_init():
    """Test AreaMaskBuilder initialization."""
    mask_builder1 = AreaMaskBuilder("nodes")
    mask_builder2 = AreaMaskBuilder("nodes", margin_radius_km=120)
    mask_builder3 = AreaMaskBuilder("nodes", mask_attr_name="mask")
    mask_builder4 = AreaMaskBuilder("nodes", margin_radius_km=120, mask_attr_name="mask")

    assert isinstance(mask_builder1, AreaMaskBuilder)
    assert isinstance(mask_builder2, AreaMaskBuilder)
    assert isinstance(mask_builder3, AreaMaskBuilder)
    assert isinstance(mask_builder4, AreaMaskBuilder)

    assert mask_builder1.reference_node_name == "nodes"
    assert mask_builder2.margin_radius_km == 120
    assert mask_builder3.mask_attr_name == "mask"
    assert mask_builder4.margin_radius_km == 120
    assert mask_builder4.mask_attr_name == "mask"


def test_chord_threshold():
    """Test AreaMaskBuilder chord threshold conversion."""
    margin_radius_km = 120
    mask_builder = AreaMaskBuilder("nodes", margin_radius_km=margin_radius_km)
    expected = float(2 * torch.sin(torch.tensor(margin_radius_km / (2 * EARTH_RADIUS), dtype=torch.float64)))

    assert mask_builder._chord_threshold == pytest.approx(expected)


@pytest.mark.parametrize("margin", [-1, 0, "120", None])
def test_fail_init_wrong_margin(margin: int):
    """Test AreaMaskBuilder initialization with invalid margin."""
    with pytest.raises(AssertionError):
        AreaMaskBuilder("nodes", margin_radius_km=margin)


@pytest.mark.parametrize("mask", [None, "mask"])
def test_fit(graph_with_nodes: HeteroData, mask: str):
    """Test AreaMaskBuilder fit."""
    mask_builder = AreaMaskBuilder("test_nodes", mask_attr_name=mask)
    assert hasattr(mask_builder, "_backend")

    mask_builder.fit(graph_with_nodes)

    assert getattr(mask_builder._backend, "_ref_vectors") is not None
    if PYG_LIB_AVAILABLE:
        assert mask_builder._backend._ref_vectors.shape[1] == 3
        assert isinstance(mask_builder._backend._ref_vectors, torch.Tensor)
    else:
        assert mask_builder._backend._kdtree is not None
        assert isinstance(mask_builder._backend._kdtree, cKDTree)


def test_fit_fail_missing_mask_attribute(graph_with_nodes: HeteroData):
    """Test AreaMaskBuilder fails when mask attribute is missing."""
    mask_builder = AreaMaskBuilder("test_nodes", mask_attr_name="wrong_mask")

    with pytest.raises(AssertionError):
        mask_builder.fit(graph_with_nodes)


def test_fit_fail(graph_with_nodes):
    """Test AreaMaskBuilder fit with wrong graph."""
    mask_builder = AreaMaskBuilder("wrong_nodes")
    with pytest.raises(AssertionError):
        mask_builder.fit(graph_with_nodes)


def test_get_mask(graph_with_nodes: HeteroData):
    """Test AreaMaskBuilder get_mask on query points that match references."""
    mask_builder = AreaMaskBuilder("test_nodes", margin_radius_km=100)
    mask_builder.fit(graph_with_nodes)

    query_coords_rad = graph_with_nodes["test_nodes"].x.cpu().numpy()
    mask = mask_builder.get_mask(query_coords_rad)

    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.bool
    assert mask.shape == (graph_with_nodes["test_nodes"].num_nodes,)
    assert mask.all()


def test_get_mask_respects_reference_mask(graph_with_nodes: HeteroData):
    """Test AreaMaskBuilder get_mask uses only masked-in references."""
    mask_builder = AreaMaskBuilder("test_nodes", margin_radius_km=100, mask_attr_name="interior_mask")
    mask_builder.fit(graph_with_nodes)

    masked_out_ref = graph_with_nodes["test_nodes"].x[0].cpu().numpy()
    masked_in_ref = graph_with_nodes["test_nodes"].x[5].cpu().numpy()
    query = np.stack([masked_out_ref, masked_in_ref], axis=0)
    mask = mask_builder.get_mask(query)

    assert mask.tolist() == [False, True]


def test_get_mask_fail_not_fitted(graph_with_nodes: HeteroData):
    """Test AreaMaskBuilder get_mask fails before fit."""
    mask_builder = AreaMaskBuilder("test_nodes")

    query_coords_rad = graph_with_nodes["test_nodes"].x.cpu().numpy()
    with pytest.raises(AssertionError):
        mask_builder.get_mask(query_coords_rad)
