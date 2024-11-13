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

from anemoi.graphs.edges import ICONTopologicalDecoderEdges
from anemoi.graphs.edges import ICONTopologicalEncoderEdges
from anemoi.graphs.edges import ICONTopologicalProcessorEdges
from anemoi.graphs.generate.icon_mesh import ICONCellDataGrid
from anemoi.graphs.generate.icon_mesh import ICONMultiMesh
from anemoi.graphs.nodes import ICONCellGridNodes
from anemoi.graphs.nodes import ICONMultimeshNodes
from anemoi.graphs.nodes import ICONNodes
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


@pytest.mark.parametrize("max_level_multimesh,max_level_dataset", [(0, 0), (0, 1), (1, 1)])
def test_init(monkeypatch, max_level_multimesh: int, max_level_dataset: int):
    """Test ICONNodes initialization."""

    monkeypatch.setattr(netCDF4, "Dataset", DatasetMock)
    node_builder = ICONNodes(
        name="test_nodes",
        grid_filename="test.nc",
        max_level_multimesh=max_level_multimesh,
        max_level_dataset=max_level_dataset,
    )
    assert isinstance(node_builder, BaseNodeBuilder)
    assert isinstance(node_builder, ICONNodes)


@pytest.mark.parametrize("Node_builder", [ICONCellGridNodes, ICONMultimeshNodes])
def test_Node_builder_dependencies(monkeypatch, Node_builder):
    """Test that the `Node_builder` depends on the presence of ICONNodes."""

    monkeypatch.setattr(netCDF4, "Dataset", DatasetMock)
    nodes = ICONNodes("test_icon_nodes", "test.nc", 0, 0)
    data_nodes = Node_builder("data_nodes", "test_icon_nodes")
    config = {}
    graph = HeteroData()
    graph = nodes.register_attributes(graph, config)

    data_nodes.update_graph(graph)

    data_nodes = ICONCellGridNodes("data_nodes2", "missing_icon_nodes")
    with pytest.raises(KeyError):
        data_nodes.update_graph(graph)


class Test_Edge_builder_dependencies:
    @pytest.fixture()
    def icon_graph(self, monkeypatch) -> HeteroData:
        """Return a HeteroData object with ICONNodes nodes."""
        graph = HeteroData()
        monkeypatch.setattr(netCDF4, "Dataset", DatasetMock)
        nodes = ICONNodes("test_icon_nodes", "test.nc", 1, 0)

        graph = nodes.update_graph(graph, {})

        data_nodes = ICONCellGridNodes("data", "test_icon_nodes")
        graph = data_nodes.register_attributes(graph, {})

        return graph

    @pytest.mark.parametrize(
        "Edge_builder", [ICONTopologicalProcessorEdges, ICONTopologicalEncoderEdges, ICONTopologicalDecoderEdges]
    )
    def test_ProcessorEdges_dependencies(self, icon_graph, Edge_builder):
        """Test that the `Edge_builder` depends on the presence of ICONNodes."""

        edges = Edge_builder(
            source_name="data",
            target_name="data",
            icon_mesh="test_icon_nodes",
        )
        edges.update_graph(icon_graph)

        edges2 = Edge_builder(
            source_name="data",
            target_name="data",
            icon_mesh="missing_icon_nodes",
        )
        with pytest.raises(KeyError):
            edges2.update_graph(icon_graph)


def test_register_nodes(monkeypatch):
    """Test ICONNodes register correctly the nodes."""
    monkeypatch.setattr(netCDF4, "Dataset", DatasetMock)
    nodes = ICONNodes("test_icon_nodes", "test.nc", 0, 0)
    graph = HeteroData()

    graph = nodes.register_nodes(graph)

    assert graph["test_icon_nodes"].x is not None
    assert isinstance(graph["test_icon_nodes"].x, torch.Tensor)
    assert graph["test_icon_nodes"].x.shape[1] == 2
    assert graph["test_icon_nodes"].x.shape[0] == 3, "number of vertices at refinement_level_v == 0"
    assert graph["test_icon_nodes"].node_type == "ICONNodes"

    nodes2 = ICONNodes("test_icon_nodes", "test.nc", 1, 0)
    graph = nodes2.register_nodes(graph)
    assert graph["test_icon_nodes"].x.shape[0] == 4, "number of vertices at refinement_level_v == 1"


def test_register_attributes(
    monkeypatch,
    graph_with_nodes: HeteroData,
):
    """Test ICONNodes register correctly the weights."""
    monkeypatch.setattr(netCDF4, "Dataset", DatasetMock)
    nodes = ICONNodes("test_icon_nodes", "test.nc", 0, 0)
    config = {"test_attr": {"_target_": "anemoi.graphs.nodes.attributes.UniformWeights"}}
    graph = HeteroData()

    graph = nodes.register_attributes(graph, config)

    assert graph["test_icon_nodes"]["_grid_filename"] is not None
    assert isinstance(graph["test_icon_nodes"]["_multi_mesh"], ICONMultiMesh)
    assert isinstance(graph["test_icon_nodes"]["_cell_grid"], ICONCellDataGrid)
