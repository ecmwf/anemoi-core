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
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.edges.builders.base import BaseEdgeBuilder
from anemoi.graphs.edges.builders.masking import NodeMaskingMixin

LOGGER = logging.getLogger(__name__)

TORCH_CLUSTER_AVAILABLE = find_spec("torch_cluster") is not None


class KNNEdges(BaseEdgeBuilder, NodeMaskingMixin):
    """Computes KNN based edges and adds them to the graph.

    Attributes
    ----------
    source_name : str
        The name of the source nodes.
    target_name : str
        The name of the target nodes.
    num_nearest_neighbours : int
        Number of nearest neighbours.
    source_mask_attr_name : str | None
        The name of the source mask attribute to filter edge connections.
    target_mask_attr_name : str | None
        The name of the target mask attribute to filter edge connections.

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
        num_nearest_neighbours: int,
        source_mask_attr_name: str | None = None,
        target_mask_attr_name: str | None = None,
    ) -> None:
        super().__init__(source_name, target_name, source_mask_attr_name, target_mask_attr_name)
        assert isinstance(num_nearest_neighbours, int), "Number of nearest neighbours must be an integer."
        assert num_nearest_neighbours > 0, "Number of nearest neighbours must be positive."
        self.num_nearest_neighbours = num_nearest_neighbours

    def _compute_edge_index_pyg(self, source_nodes: NodeStorage, target_nodes: NodeStorage) -> torch.Tensor:
        from torch_cluster.knn import knn

        source_coords, target_coords = self.get_cartesian_node_coordinates(source_nodes, target_nodes)

        edge_index = knn(source_coords, target_coords, k=self.num_nearest_neighbours)

        return torch.flip(edge_index, [0])

    def _compute_edge_index_sklearn(self, source_nodes: NodeStorage, target_nodes: NodeStorage) -> torch.Tensor:
        source_coords, target_coords = self.get_cartesian_node_coordinates(source_nodes, target_nodes)
        nearest_neighbour = NearestNeighbors(metric="euclidean", n_jobs=4)
        nearest_neighbour.fit(source_coords.cpu())
        adj_matrix = nearest_neighbour.kneighbors_graph(
            target_coords.cpu(),
            n_neighbors=self.num_nearest_neighbours,
        ).tocoo()

        # Post-process the adjacency matrix. Add masked nodes.
        adj_matrix = self.undo_masking(adj_matrix, source_nodes, target_nodes)
        edge_index = torch.from_numpy(np.stack([adj_matrix.col, adj_matrix.row], axis=0))
        return edge_index

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
        assert self.num_nearest_neighbours is not None, "number of neighbors required for knn encoder"
        LOGGER.info(
            "Using KNN-Edges (with %d nearest neighbours) between %s and %s.",
            self.num_nearest_neighbours,
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


class ReversedKNNEdges(KNNEdges):
    """
    For each target node, this builder connects its K nearest neighbors among the source nodes.

    Attributes
    ----------
    source_name : str
        The name of the source nodes.
    target_name : str
        The name of the target nodes.
    num_nearest_neighbours : int
        Number of nearest neighbours.
    source_mask_attr_name : str | None
        The name of the source mask attribute to filter edge connections.
    target_mask_attr_name : str | None
        The name of the target mask attribute to filter edge connections.

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
