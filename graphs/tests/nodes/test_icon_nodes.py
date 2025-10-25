# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import netCDF4
import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.generate.icon_mesh import ICONCellDataGrid
from anemoi.graphs.generate.icon_mesh import ICONMultiMesh
from anemoi.graphs.nodes import ICONCellGridNodes
from anemoi.graphs.nodes import ICONMultimeshNodes
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder


class DatasetMock:
    """This datasets emulates the most primitive unstructured grid with
    refinement.

    Enumeration of cells , edges and vertices in netCDF file is 1 based.
    C: cell
    E: edge
    V: vertex

    Cell C2 with its additional vertex V4 and edges E4 and E4 were added as
    a first refinement.

    [V1: 0, 1]🢀-E3--[V3: 1, 1]
      🢁      ╲             🢁
      |       ╲ [C1: ⅔, ⅔] |
      |        ╲           |
      E5        E1         E2
      |          ╲         |
      |           ╲        |
      | [C2: ⅓, ⅓] ╲       |
      |             🢆     |
    [V4: 0, 1]🢀-E4--[V2: 1, 1]

    Note: Triangular refinement does not actually work like this. This grid
    mock serves testing purposes only.

    """

    def __init__(self, *args, **kwargs):

        class MockVariable:
            def __init__(self, data, units, dimensions):
                self.data = np.ma.asarray(data)
                self.shape = data.shape
                self.units = units
                self.dimensions = dimensions

            def __getitem__(self, key):
                return self.data[key]

        self.variables = {
            "vlon": MockVariable(np.array([0, 1, 1, 0]), "radian", ("vertex",)),
            "vlat": MockVariable(np.array([1, 0, 1, 0]), "radian", ("vertex",)),
            "clon": MockVariable(np.array([0.66, 0.33]), "radian", ("cell",)),
            "clat": MockVariable(np.array([0.66, 0.33]), "radian", ("cell",)),
            "edge_vertices": MockVariable(np.array([[1, 2], [2, 3], [3, 1], [2, 4], [4, 1]]).T, "", ("nc", "edge")),
            "vertex_of_cell": MockVariable(np.array([[1, 2, 3], [1, 2, 4]]).T, "", ("nv", "cell")),
            "refinement_level_v": MockVariable(np.array([0, 0, 0, 1]), "", ("vertex",)),
            "refinement_level_c": MockVariable(np.array([0, 1]), "", ("cell",)),
        }
        """common array dimensions:
            nc: 2, # constant
            nv: 3, # constant
            vertex: 4,
            edge: 5,
            cell: 2,
        """
        self.uuidOfHGrid = "__test_data__"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.mark.parametrize("max_level", [0, 1, 2])
@pytest.mark.parametrize("node_builder_cls", [ICONMultimeshNodes, ICONCellGridNodes])
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


@pytest.mark.parametrize("node_builder_cls", [ICONCellGridNodes, ICONMultimeshNodes])
def test_node_builder_dependencies(monkeypatch, node_builder_cls: type[BaseNodeBuilder]):
    """Test that the `node_builder` depends on the presence of ICON node builders."""
    monkeypatch.setattr(netCDF4, "Dataset", DatasetMock)
    node_builder = node_builder_cls(name="data_nodes", max_level=0, grid_filename="test.nc")

    graph = HeteroData()
    graph = node_builder.update_graph(graph)

    assert isinstance(graph, HeteroData)
    assert "data_nodes" in graph.node_types


@pytest.mark.parametrize("node_builder_cls", [ICONCellGridNodes, ICONMultimeshNodes])
def test_wrong_filename(node_builder_cls: type[BaseNodeBuilder]):
    with pytest.raises(FileNotFoundError):
        node_builder_cls(name="data_nodes2", max_level=0, grid_filename="missing_icon_nodes")


def test_register_nodes(monkeypatch):
    """Test ICON node builders register correctly the nodes."""
    monkeypatch.setattr(netCDF4, "Dataset", DatasetMock)

    node_builder = ICONMultimeshNodes(name="test_icon_nodes", grid_filename="test.nc", max_level=0)

    graph = node_builder.register_nodes(HeteroData())

    assert graph["test_icon_nodes"].x is not None
    assert isinstance(graph["test_icon_nodes"].x, torch.Tensor)
    assert graph["test_icon_nodes"].x.shape[1] == 2
    assert graph["test_icon_nodes"].num_nodes == 3, "number of vertices at refinement_level_v == 0"
    assert graph["test_icon_nodes"].node_type == "ICONMultimeshNodes"

    node_builder2 = ICONMultimeshNodes(name="test_icon_nodes", grid_filename="test.nc", max_level=1)
    graph = node_builder2.register_nodes(HeteroData())
    assert graph["test_icon_nodes"].num_nodes == 4, "number of vertices at refinement_level_v == 1"
    assert graph["test_icon_nodes"].node_type == "ICONMultimeshNodes"


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
