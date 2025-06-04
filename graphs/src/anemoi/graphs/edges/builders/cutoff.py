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
import warnings
from importlib.util import find_spec

import numpy as np
import torch
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs import EARTH_RADIUS
from anemoi.graphs.edges.builders.base import BaseEdgeBuilder
from anemoi.graphs.edges.builders.masking import NodeMaskingMixin
from anemoi.graphs.utils import get_grid_reference_distance

TORCH_CLUSTER_AVAILABLE = find_spec("torch_cluster") is not None


LOGGER = logging.getLogger(__name__)


class CutOffEdges(BaseEdgeBuilder, NodeMaskingMixin):
    """Computes cut-off based edges and adds them to the graph.

    Attributes
    ----------
    source_name : str
        The name of the source nodes.
    target_name : str
        The name of the target nodes.
    cutoff_factor : float
        Factor to multiply the grid reference distance to get the cut-off radius.
    source_mask_attr_name : str | None
        The name of the source mask attribute to filter edge connections.
    target_mask_attr_name : str | None
        The name of the target mask attribute to filter edge connections.
    max_num_neighbours : int
        The maximum number of nearest neighbours to consider when building edges.

    Methods
    -------
    register_edges(graph)
        Register the edges in the graph.
    register_attributes(graph, config)
        Register attributes in the edges of the graph.
    update_graph(graph, attrs_config)
        Update the graph with the edges.
    """

    def __init__(
        self,
        source_name: str,
        target_name: str,
        cutoff_factor: float,
        source_mask_attr_name: str | None = None,
        target_mask_attr_name: str | None = None,
        max_num_neighbours: int = 64,
    ) -> None:
        super().__init__(source_name, target_name, source_mask_attr_name, target_mask_attr_name)
        assert isinstance(cutoff_factor, (int, float)), "Cutoff factor must be a float."
        assert isinstance(max_num_neighbours, int), "Number of nearest neighbours must be an integer."
        assert cutoff_factor > 0, "Cutoff factor must be positive."
        assert max_num_neighbours > 0, "Number of nearest neighbours must be positive."
        self.cutoff_factor = cutoff_factor
        self.max_num_neighbours = max_num_neighbours

    def get_cutoff_radius(self, graph: HeteroData, mask_attr: torch.Tensor | None = None) -> float:
        """Compute the cut-off radius.

        The cut-off radius is computed as the product of the target nodes
        reference distance and the cut-off factor.

        Parameters
        ----------
        graph : HeteroData
            The graph.
        mask_attr : torch.Tensor
            The mask attribute.

        Returns
        -------
        float
            The cut-off radius.
        """
        target_nodes = graph[self.target_name]
        if mask_attr is not None:
            # If masking target nodes, we have to recompute the grid reference distance only over the masked nodes
            mask = target_nodes[mask_attr]
            target_grid_reference_distance = get_grid_reference_distance(target_nodes.x, mask)
        else:
            target_grid_reference_distance = target_nodes["_grid_reference_distance"]

        radius = target_grid_reference_distance * self.cutoff_factor
        return radius

    def prepare_node_data(self, graph: HeteroData) -> tuple[NodeStorage, NodeStorage]:
        """Prepare node information and get source and target nodes."""
        self.radius = self.get_cutoff_radius(graph)
        return super().prepare_node_data(graph)

    def _compute_edge_index_pyg(self, source_nodes: NodeStorage, target_nodes: NodeStorage) -> torch.Tensor:
        from torch_cluster.radius import radius

        source_coords, target_coords = self.get_cartesian_node_coordinates(source_nodes, target_nodes)

        edge_index = radius(source_coords, target_coords, r=self.radius, max_num_neighbors=self.max_num_neighbours)

        return torch.flip(edge_index, [0])

    def _crop_to_max_num_neighbours(self, adjmat):
        """Remove neighbors exceeding the maximum allowed limit."""
        nodes_to_drop = np.maximum(np.bincount(adjmat.row) - self.max_num_neighbours, 0)
        if num_nodes_to_drop := nodes_to_drop.sum() == 0:
            return adjmat

        LOGGER.info(
            "%s is removing %d because they exceed the maximum allowed number of neighbors (%d) for each target node.",
            self.__class__.__name__,
            num_nodes_to_drop,
            self.max_num_neighbours,
        )

        # Compute indices to remove
        mask = np.ones(adjmat.nnz, dtype=bool)
        node_idx = np.where(nodes_to_drop > 0)[0]
        for node_id in node_idx:
            indices_of_largest_dist = np.argpartition(adjmat.data[adjmat.row == node_id], -nodes_to_drop[node_id])[
                -nodes_to_drop[node_id] :
            ]
            mask[np.where(adjmat.row == node_id)[0][indices_of_largest_dist]] = False

        # Define the new sparse matrix
        return coo_matrix((adjmat.data[mask], (adjmat.row[mask], adjmat.col[mask])), shape=adjmat.shape)

    def _compute_edge_index_sklearn(self, source_nodes: NodeStorage, target_nodes: NodeStorage) -> torch.Tensor:
        source_coords, target_coords = self.get_cartesian_node_coordinates(source_nodes, target_nodes)
        nearest_neighbour = NearestNeighbors(metric="euclidean", n_jobs=4)
        nearest_neighbour.fit(source_coords.cpu())
        adj_matrix = nearest_neighbour.radius_neighbors_graph(
            target_coords.cpu(), radius=self.radius, mode="distance"
        ).tocoo()

        adj_matrix = self._crop_to_max_num_neighbours(adj_matrix)
        adj_matrix = self.undo_masking(adj_matrix, source_nodes, target_nodes)
        edge_index = torch.from_numpy(np.stack([adj_matrix.col, adj_matrix.row], axis=0))

        return edge_index

    def compute_edge_index(self, source_nodes: NodeStorage, target_nodes: NodeStorage) -> torch.Tensor:
        """Get the adjacency matrix for the cut-off method.

        Parameters
        ----------
        source_nodes : NodeStorage
            The source nodes.
        target_nodes : NodeStorage
            The target nodes.

        Returns
        -------
        torch.Tensor of shape (2, num_edges)
            The adjacency matrix.
        """
        LOGGER.info(
            "Using CutOff-Edges (with radius = %.1f km) between %s and %s.",
            self.radius * EARTH_RADIUS,
            self.source_name,
            self.target_name,
        )

        if TORCH_CLUSTER_AVAILABLE:
            edge_index = self._compute_edge_index_pyg(source_nodes, target_nodes)
        else:
            warnings.warn(
                "The 'torch-cluster' library is not installed. Installing 'torch-cluster' can significantly improve "
                "performance for graph creation. You can install it using 'pip install torch-cluster'.",
                UserWarning,
            )

            edge_index = self._compute_edge_index_sklearn(source_nodes, target_nodes)

        return edge_index


class ReversedCutOffEdges(CutOffEdges):
    """For each target node, this builder connects all source nodes within the cutoff_factor.

    Attributes
    ----------
    source_name : str
        The name of the source nodes.
    target_name : str
        The name of the target nodes.
    cutoff_factor : float
        Factor to multiply the grid reference distance to get the cut-off radius.
    source_mask_attr_name : str | None
        The name of the source mask attribute to filter edge connections.
    target_mask_attr_name : str | None
        The name of the target mask attribute to filter edge connections.
    max_num_neighbours : int
        The maximum number of nearest neighbours to consider when building edges.

    Methods
    -------
    register_edges(graph)
        Register the edges in the graph.
    register_attributes(graph, config)
        Register attributes in the edges of the graph.
    update_graph(graph, attrs_config)
        Update the graph with the edges.
    """

    def compute_edge_index(self, source_nodes: NodeStorage, target_nodes: NodeStorage) -> torch.Tensor:
        return super().compute_edge_index(target_nodes, source_nodes)
