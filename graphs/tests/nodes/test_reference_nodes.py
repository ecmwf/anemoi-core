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
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.graphs.create import GraphCreator
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder
from anemoi.graphs.nodes.builders.from_reference import ReferenceNodes


def test_init():
    node_builder = ReferenceNodes(reference_node_name="data", name="smooth")
    assert isinstance(node_builder, BaseNodeBuilder)
    assert isinstance(node_builder, ReferenceNodes)


def test_register_nodes_copies_coordinates():
    graph = HeteroData()
    graph["data"].x = torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=torch.float64)

    node_builder = ReferenceNodes(reference_node_name="data", name="smooth")
    graph = node_builder.register_nodes(graph)

    assert graph["smooth"].x is not None
    assert isinstance(graph["smooth"].x, torch.Tensor)
    assert graph["smooth"].x.dtype == torch.float32
    assert torch.equal(graph["smooth"].x, graph["data"].x.float())
    assert graph["smooth"].node_type == "ReferenceNodes"
    assert "_grid_reference_distance" in graph["smooth"]

    graph["data"].x[0, 0] = -999.0
    assert graph["smooth"].x[0, 0] == 10.0


def test_register_nodes_missing_reference():
    graph = HeteroData()
    node_builder = ReferenceNodes(reference_node_name="data", name="smooth")
    with pytest.raises(AssertionError, match="must be registered before"):
        node_builder.register_nodes(graph)


def test_register_nodes_reference_without_coordinates():
    graph = HeteroData()
    graph["data"].node_type = "NoCoords"
    node_builder = ReferenceNodes(reference_node_name="data", name="smooth")
    with pytest.raises(AssertionError, match="does not have coordinates"):
        node_builder.register_nodes(graph)


def test_graph_creator_reference_nodes_build_in_order():
    config = {
        "nodes": {
            "data": {
                "node_builder": {
                    "_target_": "anemoi.graphs.nodes.LatLonNodes",
                    "latitudes": [-10.0, 0.0, 10.0],
                    "longitudes": [0.0, 120.0, 240.0],
                },
            },
            "smooth": {
                "node_builder": {
                    "_target_": "anemoi.graphs.nodes.ReferenceNodes",
                    "reference_node_name": "data",
                },
            },
        },
        "edges": [],
    }

    graph = GraphCreator(config=OmegaConf.create(config)).create(save_path=None)
    assert torch.equal(graph["smooth"].x, graph["data"].x)


def test_graph_creator_reference_nodes_fails_when_reference_built_late():
    config = {
        "nodes": {
            "smooth": {
                "node_builder": {
                    "_target_": "anemoi.graphs.nodes.ReferenceNodes",
                    "reference_node_name": "data",
                },
            },
            "data": {
                "node_builder": {
                    "_target_": "anemoi.graphs.nodes.LatLonNodes",
                    "latitudes": [-10.0, 0.0, 10.0],
                    "longitudes": [0.0, 120.0, 240.0],
                },
            },
        },
        "edges": [],
    }

    with pytest.raises(AssertionError, match="must be registered before"):
        GraphCreator(config=OmegaConf.create(config)).create(save_path=None)
