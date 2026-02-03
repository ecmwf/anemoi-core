# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import torch
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.edges.builders.base import BaseEdgeBuilder
from anemoi.graphs.generate.icon_mesh import ICONCellDataGrid
from anemoi.graphs.generate.icon_mesh import ICONMultiMesh

LOGGER = logging.getLogger(__name__)


class ICONTopologicalProcessorEdges(BaseEdgeBuilder):
    """ICON Topological Processor Edges

    Computes edges based on ICON grid topology: processor grid built
    from ICON grid vertices.
    """

    def compute_edge_index(self, source_nodes: NodeStorage, target_nodes: NodeStorage) -> torch.Tensor:
        """Compute the edge indices for the KNN method.

        Parameters
        ----------
        source_nodes : NodeStorage
            The source nodes.
        target_nodes : NodeStorage
            The target nodes.

        Returns
        -------
        torch.Tensor of shape (2, num_edges)
            Indices of source and target nodes connected by an edge.
        """
        assert isinstance(
            source_nodes["_icon_nodes"], ICONMultiMesh
        ), f"{self.__class__.__name__}: source nodes must be ICONMultiMesh"
        assert isinstance(
            target_nodes["_icon_nodes"], ICONMultiMesh
        ), f"{self.__class__.__name__}: target nodes must be ICONMultiMesh"
        edge_index = source_nodes["_icon_nodes"].multi_mesh_edges
        return torch.from_numpy(edge_index.T)


class BaseICONEdgeBuilder(BaseEdgeBuilder):
    """Base ICON Edge Builder."""

    def prepare_node_data(self, graph: HeteroData) -> tuple[ICONCellDataGrid, ICONMultiMesh]:
        nodes_names = self.source_name, self.target_name
        cell_grid = graph[nodes_names[self.vertex_index[0]]]["_icon_nodes"]
        assert isinstance(
            cell_grid, ICONCellDataGrid
        ), f"{self.__class__.__name__}: source nodes must be ICONCellDataGrid"
        multi_mesh = graph[nodes_names[self.vertex_index[1]]]["_icon_nodes"]
        assert isinstance(multi_mesh, ICONMultiMesh), f"{self.__class__.__name__}: target nodes must be ICONMultiMesh"
        return cell_grid, multi_mesh

    def compute_edge_index(self, cell_grid: ICONCellDataGrid, multi_mesh: ICONMultiMesh) -> torch.Tensor:
        edge_vertices = cell_grid.get_grid2mesh_edges(multi_mesh)
        return torch.from_numpy(edge_vertices[:, self.vertex_index].T)


class ICONTopologicalEncoderEdges(BaseICONEdgeBuilder):
    """ICON Topological Encoder Edges

    Computes encoder edges based on ICON grid topology: ICON cell
    circumcenters for mapped onto processor grid built from ICON grid
    vertices.
    """

    vertex_index: tuple[int, int] = (0, 1)


class ICONTopologicalDecoderEdges(BaseICONEdgeBuilder):
    """ICON Topological Decoder Edges

    Computes decoder edges based on ICON grid topology: mapping from
    processor grid built from ICON grid vertices onto ICON cell
    circumcenters.
    """

    vertex_index: tuple[int, int] = (1, 0)
