# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging

import numpy as np
import scipy
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.edges.builders.base import BaseEdgeBuilder
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class ICONTopologicalBaseEdgeBuilder(BaseEdgeBuilder):
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

    def __init__(
        self,
        source_name: str,
        target_name: str,
        icon_mesh: str,
        source_mask_attr_name: str | None = None,
        target_mask_attr_name: str | None = None,
    ):
        self.icon_mesh = icon_mesh
        super().__init__(source_name, target_name, source_mask_attr_name, target_mask_attr_name)

    def update_graph(self, graph: HeteroData, attrs_config: DotDict = None) -> HeteroData:
        """Update the graph with the edges."""
        assert self.icon_mesh is not None, f"{self.__class__.__name__} requires initialized icon_mesh."
        self.icon_sub_graph = graph[self.icon_mesh][self.sub_graph_address]
        return super().update_graph(graph, attrs_config)

    def get_adjacency_matrix(self, source_nodes: NodeStorage, target_nodes: NodeStorage):
        """Parameters
        ----------
        source_nodes : NodeStorage
            The source nodes.
        target_nodes : NodeStorage
            The target nodes.
        """
        LOGGER.info(f"Using ICON topology {self.source_name}>{self.target_name}")
        nrows = self.icon_sub_graph.num_edges
        adj_matrix = scipy.sparse.coo_matrix(
            (
                np.ones(nrows),
                (
                    self.icon_sub_graph.edge_vertices[:, self.vertex_index[0]],
                    self.icon_sub_graph.edge_vertices[:, self.vertex_index[1]],
                ),
            )
        )
        return adj_matrix


class ICONTopologicalProcessorEdges(ICONTopologicalBaseEdgeBuilder):
    """Computes edges based on ICON grid topology: processor grid built
    from ICON grid vertices.
    """

    def __init__(
        self,
        source_name: str,
        target_name: str,
        icon_mesh: str,
        source_mask_attr_name: str | None = None,
        target_mask_attr_name: str | None = None,
    ):
        self.sub_graph_address = "_multi_mesh"
        self.vertex_index = (1, 0)
        super().__init__(
            source_name,
            target_name,
            icon_mesh,
            source_mask_attr_name,
            target_mask_attr_name,
        )


class ICONTopologicalEncoderEdges(ICONTopologicalBaseEdgeBuilder):
    """Computes encoder edges based on ICON grid topology: ICON cell
    circumcenters for mapped onto processor grid built from ICON grid
    vertices.
    """

    def __init__(
        self,
        source_name: str,
        target_name: str,
        icon_mesh: str,
        source_mask_attr_name: str | None = None,
        target_mask_attr_name: str | None = None,
    ):
        self.sub_graph_address = "_cell_grid"
        self.vertex_index = (1, 0)
        super().__init__(
            source_name,
            target_name,
            icon_mesh,
            source_mask_attr_name,
            target_mask_attr_name,
        )


class ICONTopologicalDecoderEdges(ICONTopologicalBaseEdgeBuilder):
    """Computes encoder edges based on ICON grid topology: mapping from
    processor grid built from ICON grid vertices onto ICON cell
    circumcenters.
    """

    def __init__(
        self,
        source_name: str,
        target_name: str,
        icon_mesh: str,
        source_mask_attr_name: str | None = None,
        target_mask_attr_name: str | None = None,
    ):
        self.sub_graph_address = "_cell_grid"
        self.vertex_index = (0, 1)
        super().__init__(
            source_name,
            target_name,
            icon_mesh,
            source_mask_attr_name,
            target_mask_attr_name,
        )
