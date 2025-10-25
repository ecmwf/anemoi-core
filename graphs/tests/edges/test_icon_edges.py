# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import netCDF4
import pytest
from torch_geometric.data import HeteroData

from anemoi.graphs.edges import ICONTopologicalDecoderEdges
from anemoi.graphs.edges import ICONTopologicalEncoderEdges
from anemoi.graphs.edges import ICONTopologicalProcessorEdges
from anemoi.graphs.nodes import ICONCellGridNodes
from anemoi.graphs.nodes import ICONMultiMeshNodes


class TestEdgeBuilderDependencies:
    """Test that ICON edge builders depend on the presence of ICON node builders."""

    @pytest.fixture()
    def icon_graph(self, monkeypatch, icon_dataset_mock) -> HeteroData:
        """Return a HeteroData object with ICON node builders."""
        monkeypatch.setattr(netCDF4, "Dataset", icon_dataset_mock)
        data_node_builder = ICONCellGridNodes(name="data", grid_filename="test.nc", max_level=1)
        hidden_node_builder = ICONMultiMeshNodes(name="hidden", grid_filename="test.nc", max_level=1)

        graph = HeteroData()
        graph = data_node_builder.update_graph(graph, {})
        graph = hidden_node_builder.update_graph(graph, {})

        return graph

    def test_encoder(self, icon_graph: HeteroData):
        """Test that the `ICONTopologicalEncoderEdges` depends on the presence of ICON node builders."""
        edge_builder = ICONTopologicalEncoderEdges(source_name="data", target_name="hidden")
        assert ("data", "to", "hidden") not in icon_graph.edge_types

        icon_graph = edge_builder.update_graph(icon_graph)

        assert ("data", "to", "hidden") in icon_graph.edge_types
        assert hasattr(icon_graph["data", "to", "hidden"], "edge_index")

    def test_wrong_encoder(self, icon_graph: HeteroData):
        """Test that the `ICONTopologicalEncoderEdges` depends on the presence of ICON node builders."""
        edge_builder = ICONTopologicalEncoderEdges(source_name="hidden", target_name="data")

        with pytest.raises(AssertionError):
            edge_builder.update_graph(icon_graph)

    def test_decoder(self, icon_graph: HeteroData):
        """Test that the `ICONTopologicalDecoderEdges` depends on the presence of ICON node builders."""
        edge_builder = ICONTopologicalDecoderEdges(source_name="hidden", target_name="data")
        assert ("hidden", "to", "data") not in icon_graph.edge_types

        icon_graph = edge_builder.update_graph(icon_graph)

        assert ("hidden", "to", "data") in icon_graph.edge_types
        assert hasattr(icon_graph["hidden", "to", "data"], "edge_index")

    def test_wrong_decoder(self, icon_graph: HeteroData):
        """Test that the `ICONTopologicalDecoderEdges` depends on the presence of ICON node builders."""
        edge_builder = ICONTopologicalDecoderEdges(source_name="data", target_name="hidden")

        with pytest.raises(AssertionError):
            edge_builder.update_graph(icon_graph)

    def test_processor(self, icon_graph: HeteroData):
        """Test that the `ICONTopologicalProcessorEdges` depends on the presence of ICON node builders."""
        edge_builder = ICONTopologicalProcessorEdges(source_name="hidden", target_name="hidden")
        assert ("hidden", "to", "hidden") not in icon_graph.edge_types

        icon_graph = edge_builder.update_graph(icon_graph)

        assert ("hidden", "to", "hidden") in icon_graph.edge_types
        assert hasattr(icon_graph["hidden", "to", "hidden"], "edge_index")

    def test_wrong_processor(self, icon_graph: HeteroData):
        """Test that the `ICONTopologicalProcessorEdges` depends on the presence of ICON node builders."""
        edge_builder = ICONTopologicalProcessorEdges(source_name="data", target_name="hidden")

        with pytest.raises(AssertionError):
            edge_builder.update_graph(icon_graph)
