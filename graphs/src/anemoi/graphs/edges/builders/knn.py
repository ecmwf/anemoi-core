# (C) Copyright 2024 Anemoi contributors.
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
