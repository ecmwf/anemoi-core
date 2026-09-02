# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import netCDF4
import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.generate.icon_mesh import ICONCellDataGrid
from anemoi.graphs.generate.icon_mesh import ICONMultiMesh
from anemoi.graphs.nodes import ICONCellGridNodes
from anemoi.graphs.nodes import ICONMultiMeshNodes
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder
from anemoi.graphs.testing import DatasetMock


@pytest.mark.parametrize("max_level", [0, 1, 2])
@pytest.mark.parametrize("node_builder_cls", [ICONMultiMeshNodes, ICONCellGridNodes])
def test_init(monkeypatch, max_level: int, node_builder_cls: type[BaseNodeBuilder]):
    """Test ICON node builders initialization."""

    monkeypatch.setattr(netCDF4, "Dataset", DatasetMock)
    node_builder = node_builder_cls(
        name="test_nodes",
        grid_filename="test.nc",
        max_level=max_level,
    )
    assert isinstance(node_builder, BaseNodeBuilder)
    assert hasattr(node_builder, "icon_nodes")
    assert isinstance(node_builder.icon_nodes, (ICONMultiMesh, ICONCellDataGrid))
    assert hasattr(node_builder.icon_nodes, "nodeset")
    assert hasattr(node_builder.icon_nodes.nodeset, "gc_vertices")


@pytest.mark.parametrize("node_builder_cls", [ICONCellGridNodes, ICONMultiMeshNodes])
def test_node_builder_dependencies(monkeypatch, node_builder_cls: type[BaseNodeBuilder]):
    """Test that the `node_builder` depends on the presence of ICON node builders."""
    monkeypatch.setattr(netCDF4, "Dataset", DatasetMock)
    node_builder = node_builder_cls(name="data_nodes", max_level=0, grid_filename="test.nc")

    graph = HeteroData()
    graph = node_builder.update_graph(graph)

    assert isinstance(graph, HeteroData)
    assert "data_nodes" in graph.node_types


@pytest.mark.parametrize("node_builder_cls", [ICONCellGridNodes, ICONMultiMeshNodes])
def test_wrong_filename(node_builder_cls: type[BaseNodeBuilder]):
    with pytest.raises(FileNotFoundError):
        node_builder_cls(name="data_nodes2", max_level=0, grid_filename="missing_icon_nodes")


def test_register_nodes(monkeypatch):
    """Test ICON node builders register correctly the nodes."""
    monkeypatch.setattr(netCDF4, "Dataset", DatasetMock)

    node_builder = ICONMultiMeshNodes(name="test_icon_nodes", grid_filename="test.nc", max_level=0)

    graph = node_builder.register_nodes(HeteroData())

    assert graph["test_icon_nodes"].x is not None
    assert isinstance(graph["test_icon_nodes"].x, torch.Tensor)
    assert graph["test_icon_nodes"].x.shape[1] == 2
    assert graph["test_icon_nodes"].num_nodes == 3, "number of vertices at refinement_level_v == 0"
    assert graph["test_icon_nodes"].node_type == "ICONMultiMeshNodes"

    node_builder2 = ICONMultiMeshNodes(name="test_icon_nodes", grid_filename="test.nc", max_level=1)
    graph = node_builder2.register_nodes(HeteroData())
    assert graph["test_icon_nodes"].num_nodes == 4, "number of vertices at refinement_level_v == 1"
    assert graph["test_icon_nodes"].node_type == "ICONMultiMeshNodes"


def test_register_attributes(
    monkeypatch,
    graph_with_nodes: HeteroData,
):
    """Test ICONNodes register correctly the weights."""
    monkeypatch.setattr(netCDF4, "Dataset", DatasetMock)
    nodes = ICONCellGridNodes(name="test_nodes", max_level=0, grid_filename="test.nc")
    config = {"test_attr": {"_target_": "anemoi.graphs.nodes.attributes.UniformWeights"}}

    graph = nodes.register_attributes(graph_with_nodes, config)

    assert "test_attr" in graph["test_nodes"]
    assert torch.mean(graph["test_nodes"].test_attr) == 1.0
    assert isinstance(graph["test_nodes"]["_icon_nodes"], ICONCellDataGrid)
    assert hasattr(graph["test_nodes"]["_icon_nodes"], "grid_filename")
