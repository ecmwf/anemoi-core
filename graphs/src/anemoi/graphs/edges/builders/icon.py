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
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.edges.builders.base import BaseEdgeBuilder

LOGGER = logging.getLogger(__name__)


class ICONTopologicalProcessorEdges(BaseEdgeBuilder):
    """ICON Topological Processor Edges

    Computes edges based on ICON grid topology: processor grid built
    from ICON grid vertices.
    """

    def compute_edge_index(self, source_nodes: NodeStorage, _target_nodes: NodeStorage) -> torch.Tensor:
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
        edge_index = source_nodes["_icon_nodes"].edge_vertices.T
        return torch.from_numpy(edge_index)


class ICONTopologicalEncoderEdges(BaseEdgeBuilder):
    """ICON Topological Encoder Edges

    Computes encoder edges based on ICON grid topology: ICON cell
    circumcenters for mapped onto processor grid built from ICON grid
    vertices.
    """

    vertex_index: tuple[int, int] = (0, 1)
    
    def compute_edge_index(self, source_nodes: NodeStorage, target_nodes: NodeStorage) -> torch.Tensor:
        cell_grid = source_nodes["_icon_nodes"]
        multi_mesh = target_nodes["_icon_nodes"]
        edge_vertices = cell_grid.get_grid2mesh_edges(multi_mesh)
        return torch.from_numpy(edge_vertices[:, self.vertex_index].T)


class ICONTopologicalDecoderEdges(BaseEdgeBuilder):
    """ICON Topological Decoder Edges

    Computes encoder edges based on ICON grid topology: mapping from
    processor grid built from ICON grid vertices onto ICON cell
    circumcenters.
    """

    vertex_index: tuple[int, int] = (1, 0)

    def compute_edge_index(self, source_nodes: NodeStorage, target_nodes: NodeStorage) -> torch.Tensor:
        cell_grid = target_nodes["_icon_nodes"]
        multi_mesh = source_nodes["_icon_nodes"]
        edge_vertices = cell_grid.get_grid2mesh_edges(multi_mesh)
        return torch.from_numpy(edge_vertices[:, self.vertex_index].T)
