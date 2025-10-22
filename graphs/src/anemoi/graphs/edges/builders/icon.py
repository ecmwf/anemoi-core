# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import ABC

import torch
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.edges.builders.base import BaseEdgeBuilder
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class ICONTopologicalBaseEdgeBuilder(BaseEdgeBuilder, ABC):
    """Base class for computing edges based on ICON grid topology.

    Attributes
    ----------
    source_name : str
        The name of the source nodes.
    target_name : str
        The name of the target nodes.
    icon_mesh   : str
        The name of the ICON mesh (defines both the processor mesh and the data)
    """

    vertex_index: tuple[int, int]
    sub_graph_address: str

    def compute_edge_index(self, _source_nodes: NodeStorage, target_nodes: NodeStorage) -> torch.Tensor:
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
        edge_index = self.icon_sub_graph.edge_vertices[:, self.vertex_index].T
        return torch.from_numpy(edge_index)


class ICONTopologicalProcessorEdges(ICONTopologicalBaseEdgeBuilder):
    """ICON Topological Processor Edges

    Computes edges based on ICON grid topology: processor grid built
    from ICON grid vertices.
    """

    vertex_index: tuple[int, int] = (0, 1)
    sub_graph_address: str = "_multi_mesh"

    def update_graph(self, graph: HeteroData, attrs_config: DotDict | None = None) -> HeteroData:
        self.icon_sub_graph = graph[self.source_name][self.sub_graph_address]
        return super().update_graph(graph, attrs_config)


class ICONTopologicalEncoderEdges(ICONTopologicalBaseEdgeBuilder):
    """ICON Topological Encoder Edges

    Computes encoder edges based on ICON grid topology: ICON cell
    circumcenters for mapped onto processor grid built from ICON grid
    vertices.
    """

    vertex_index: tuple[int, int] = (0, 1)
    sub_graph_address: str = "_cell_grid"
    
    def compute_edge_index(self, _source_nodes: NodeStorage, target_nodes: NodeStorage) -> torch.Tensor:
        edge_vertices = self.icon_sub_graph._get_grid2mesh_edges(self.multi_mesh)
        return torch.from_numpy(edge_vertices[:, self.vertex_index].T)

    def update_graph(self, graph: HeteroData, attrs_config: DotDict | None = None) -> HeteroData:
        self.icon_sub_graph = graph[self.source_name][self.sub_graph_address]
        self.multi_mesh = graph[self.target_name]["_multi_mesh"]
        return super().update_graph(graph, attrs_config)


class ICONTopologicalDecoderEdges(ICONTopologicalBaseEdgeBuilder):
    """ICON Topological Decoder Edges

    Computes encoder edges based on ICON grid topology: mapping from
    processor grid built from ICON grid vertices onto ICON cell
    circumcenters.
    """

    vertex_index: tuple[int, int] = (1, 0)
    sub_graph_address: str = "_cell_grid"

    def compute_edge_index(self, _source_nodes: NodeStorage, target_nodes: NodeStorage) -> torch.Tensor:
        edge_vertices = self.icon_sub_graph._get_grid2mesh_edges(self.multi_mesh)
        return torch.from_numpy(edge_vertices[:, self.vertex_index].T)

    def update_graph(self, graph: HeteroData, attrs_config: DotDict | None = None) -> HeteroData:
        self.icon_sub_graph = graph[self.target_name][self.sub_graph_address]
        self.multi_mesh = graph[self.source_name]["_multi_mesh"]
        return super().update_graph(graph, attrs_config)
