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

from anemoi.graphs.nodes.attributes import SphericalAreaWeights
from anemoi.graphs.nodes.attributes import UniformWeights
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder
from anemoi.graphs.nodes.builders.from_healpix import HEALPixNodes


@pytest.mark.parametrize("resolution", [2, 5, 7])
def test_init(resolution: int):
    """Test HEALPixNodes initialization."""
    node_builder = HEALPixNodes(resolution, "test_nodes")
    assert isinstance(node_builder, BaseNodeBuilder)
    assert isinstance(node_builder, HEALPixNodes)


@pytest.mark.parametrize("resolution", ["2", 4.3, -7])
def test_fail_init(resolution: int):
    """Test HEALPixNodes initialization with invalid resolution."""
    with pytest.raises(AssertionError):
        HEALPixNodes(resolution, "test_nodes")


@pytest.mark.parametrize("resolution", [2, 5, 7])
def test_register_nodes(resolution: int):
    """Test HEALPixNodes register correctly the nodes."""
    node_builder = HEALPixNodes(resolution, "test_nodes")
    graph = HeteroData()

    graph = node_builder.register_nodes(graph)

    assert graph["test_nodes"].x is not None
    assert isinstance(graph["test_nodes"].x, torch.Tensor)
    assert graph["test_nodes"].x.shape[1] == 2
    assert graph["test_nodes"].node_type == "HEALPixNodes"


@pytest.mark.parametrize("resolution", [2, 5])
def test_nest_ordering(resolution: int):
    """nest_ordering selects NEST vs RING pixel ordering; defaults to NEST."""
    nest = HEALPixNodes(resolution, "test_nodes", nest_ordering=True).get_coordinates()
    ring = HEALPixNodes(resolution, "test_nodes", nest_ordering=False).get_coordinates()
    default = HEALPixNodes(resolution, "test_nodes").get_coordinates()

    # default is NEST, backward-compatible with the previously hardcoded nest=True
    assert torch.equal(default, nest)

    # same set of pixel centres, different row order
    assert not torch.equal(nest, ring)
    nest_sorted = nest[nest[:, 0].argsort(stable=True)].sort(dim=0).values
    ring_sorted = ring[ring[:, 0].argsort(stable=True)].sort(dim=0).values
    assert torch.allclose(nest_sorted, ring_sorted)

    # RING ordering is isolatitude, sorted north -> south
    assert torch.all(torch.diff(ring[:, 0]) <= 1e-9)
    assert not torch.all(torch.diff(nest[:, 0]) <= 1e-9)


@pytest.mark.parametrize("attr_class", [UniformWeights, SphericalAreaWeights])
@pytest.mark.parametrize("resolution", [2, 5, 7])
def test_register_attributes(graph_with_nodes: HeteroData, attr_class, resolution: int):
    """Test HEALPixNodes register correctly the weights."""
    node_builder = HEALPixNodes(resolution, "test_nodes")
    config = {"test_attr": {"_target_": f"anemoi.graphs.nodes.attributes.{attr_class.__name__}"}}

    graph = node_builder.register_attributes(graph_with_nodes, config)

    assert graph["test_nodes"]["test_attr"] is not None
    assert isinstance(graph["test_nodes"]["test_attr"], torch.Tensor)
    assert graph["test_nodes"]["test_attr"].shape[0] == graph["test_nodes"].x.shape[0]
