# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data.storage import NodeStorage
from torch_geometric.nn import knn

from anemoi.graphs.edges.builders.base import BaseDistanceEdgeBuilders
from anemoi.graphs.utils import intersect_edges

LOGGER = logging.getLogger(__name__)


class KNNEdges(BaseDistanceEdgeBuilders):
    """Computes KNN based edges and adds them to the graph.

    It uses as reference the target nodes.

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

        LOGGER.info(
            "Using KNN-Edges (with %d nearest neighbours) between %s and %s.",
            self.num_nearest_neighbours,
            self.source_name,
            self.target_name,
        )

    def _compute_edge_index_pyg(self, source_coords: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
        edge_index = knn(source_coords, target_coords, k=self.num_nearest_neighbours)
        edge_index = torch.flip(edge_index, [0])
        return edge_index

    def _compute_adj_matrix_sklearn(self, source_coords: torch.Tensor, target_coords: torch.Tensor) -> np.ndarray:
        nearest_neighbour = NearestNeighbors(metric="euclidean", n_jobs=4)
        nearest_neighbour.fit(source_coords.cpu())
        adj_matrix = nearest_neighbour.kneighbors_graph(
            target_coords.cpu(),
            n_neighbors=self.num_nearest_neighbours,
        ).tocoo()

        return adj_matrix


class ReversedKNNEdges(KNNEdges):
    """Computes KNN based edges and adds them to the graph.

    It uses as reference the source nodes.

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

    def get_cartesian_node_coordinates(
        self, source_nodes: NodeStorage, target_nodes: NodeStorage
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_coords, target_coords = super().get_cartesian_node_coordinates(source_nodes, target_nodes)
        return target_coords, source_coords

    def undo_masking_adj_matrix(self, adj_matrix, source_nodes: NodeStorage, target_nodes: NodeStorage):
        adj_matrix = adj_matrix.T
        return super().undo_masking_adj_matrix(adj_matrix, source_nodes, target_nodes)

    def undo_masking_edge_index(
        self, edge_index: torch.Tensor, source_nodes: NodeStorage, target_nodes: NodeStorage
    ) -> torch.Tensor:
        edge_index = torch.flip(edge_index, [0])
        return super().undo_masking_edge_index(edge_index, source_nodes, target_nodes)


class MutualKNNEdges(BaseDistanceEdgeBuilders):
    """Computes mutual KNN based edges and adds them to the graph.

    An edge between a source node and a target node is kept only if the target
    node is among the source node's nearest target nodes *and* the source node
    is among the target node's nearest source nodes. The result is the
    intersection of the ``KNNEdges`` and ``ReversedKNNEdges`` edge sets.

    This tapers connectivity symmetrically across a resolution transition (for
    example the global/regional boundary of a stretched grid) and bounds node
    degree on both sides at once. ``num_nearest_neighbours`` and
    ``reversed_num_nearest_neighbours`` may differ to control fan-in and
    fan-out independently.

    Attributes
    ----------
    source_name : str
        The name of the source nodes.
    target_name : str
        The name of the target nodes.
    num_nearest_neighbours : int
        Number of nearest source nodes considered for each target node.
    reversed_num_nearest_neighbours : int
        Number of nearest target nodes considered for each source node.
        Defaults to ``num_nearest_neighbours`` when not provided.
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
        reversed_num_nearest_neighbours: int | None = None,
        source_mask_attr_name: str | None = None,
        target_mask_attr_name: str | None = None,
    ) -> None:
        super().__init__(source_name, target_name, source_mask_attr_name, target_mask_attr_name)
        assert isinstance(num_nearest_neighbours, int), "Number of nearest neighbours must be an integer."
        assert num_nearest_neighbours > 0, "Number of nearest neighbours must be positive."

        if reversed_num_nearest_neighbours is None:
            reversed_num_nearest_neighbours = num_nearest_neighbours
        assert isinstance(
            reversed_num_nearest_neighbours, int
        ), "Reversed number of nearest neighbours must be an integer."
        assert reversed_num_nearest_neighbours > 0, "Reversed number of nearest neighbours must be positive."

        self.num_nearest_neighbours = num_nearest_neighbours
        self.reversed_num_nearest_neighbours = reversed_num_nearest_neighbours

        LOGGER.info(
            "Using MutualKNN-Edges (forward k=%d, reversed k=%d) between %s and %s.",
            self.num_nearest_neighbours,
            self.reversed_num_nearest_neighbours,
            self.source_name,
            self.target_name,
        )

    def _compute_edge_index_pyg(self, source_coords: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
        # Forward: for each target node, its nearest source nodes.
        # knn(x=source, y=target) -> rows (target_idx, source_idx); flip -> (source, target).
        forward = torch.flip(knn(source_coords, target_coords, k=self.num_nearest_neighbours), [0])
        # Reversed: for each source node, its nearest target nodes.
        # knn(x=target, y=source) -> rows (source_idx, target_idx); already (source, target).
        reversed_ = knn(target_coords, source_coords, k=self.reversed_num_nearest_neighbours)
        return intersect_edges(forward, reversed_)

    def _compute_adj_matrix_sklearn(self, source_coords: torch.Tensor, target_coords: torch.Tensor) -> np.ndarray:
        # Forward adjacency: rows = target, cols = source.
        forward_nn = NearestNeighbors(metric="euclidean", n_jobs=4)
        forward_nn.fit(source_coords.cpu())
        forward = forward_nn.kneighbors_graph(target_coords.cpu(), n_neighbors=self.num_nearest_neighbours).tocsr()

        # Reversed adjacency: rows = source, cols = target.
        reversed_nn = NearestNeighbors(metric="euclidean", n_jobs=4)
        reversed_nn.fit(target_coords.cpu())
        reversed_ = reversed_nn.kneighbors_graph(
            source_coords.cpu(), n_neighbors=self.reversed_num_nearest_neighbours
        ).tocsr()

        # Keep only edges present in both directions (transpose reversed_ to rows = target).
        mutual = forward.multiply(reversed_.T)
        return mutual.tocoo()
