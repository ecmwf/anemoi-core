import logging
from abc import ABC
from abc import abstractmethod
from typing import Optional

import numpy as np
import torch
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs import EARTH_RADIUS
from anemoi.graphs.utils import get_grid_reference_distance

LOGGER = logging.getLogger(__name__)


class BaseEdgeBuilder(ABC):
    """Base class for edge builders."""

    def __init__(self, source_name: str, target_name: str):
        self.source_name = source_name
        self.target_name = target_name

    @property
    def name(self) -> tuple[str, str, str]:
        """Name of the edge subgraph."""
        return self.source_name, "to", self.target_name

    @abstractmethod
    def get_adjacency_matrix(self, source_nodes: NodeStorage, target_nodes: NodeStorage): ...

    def prepare_node_data(self, graph: HeteroData) -> tuple[NodeStorage, NodeStorage]:
        """Prepare nodes information."""
        return graph[self.source_name], graph[self.target_name]

    def get_edge_index(self, graph: HeteroData) -> torch.Tensor:
        """Get the edge indices of source and target nodes.

        Parameters
        ----------
        graph : HeteroData
            The graph.

        Returns
        -------
        torch.Tensor of shape (2, num_edges)
            The edge indices.
        """
        source_nodes, target_nodes = self.prepare_node_data(graph)

        adjmat = self.get_adjacency_matrix(source_nodes, target_nodes)

        # Get source & target indices of the edges
        edge_index = np.stack([adjmat.col, adjmat.row], axis=0)

        return torch.from_numpy(edge_index).to(torch.int32)

    def register_edges(self, graph: HeteroData) -> HeteroData:
        """Register edges in the graph.

        Parameters
        ----------
        graph : HeteroData
            The graph to register the edges.

        Returns
        -------
        HeteroData
            The graph with the registered edges.
        """
        graph[self.name].edge_index = self.get_edge_index(graph)
        graph[self.name].edge_type = type(self).__name__
        return graph

    def register_attributes(self, graph: HeteroData, config: DotDict) -> HeteroData:
        """Register attributes in the edges of the graph specified.

        Parameters
        ----------
        graph : HeteroData
            The graph to register the attributes.
        config : DotDict
            The configuration of the attributes.

        Returns
        -------
        HeteroData
            The graph with the registered attributes.
        """
        for attr_name, attr_config in config.items():
            graph[self.name][attr_name] = instantiate(attr_config).compute(graph, self.name)
        return graph

    def update_graph(self, graph: HeteroData, attrs_config: Optional[DotDict] = None) -> HeteroData:
        """Update the graph with the edges.

        Parameters
        ----------
        graph : HeteroData
            The graph.
        attrs_config : DotDict
            The configuration of the edge attributes.

        Returns
        -------
        HeteroData
            The graph with the edges.
        """
        graph = self.register_edges(graph)

        if attrs_config is None:
            return graph

        graph = self.register_attributes(graph, attrs_config)

        return graph


class KNNEdges(BaseEdgeBuilder):
    """Computes KNN based edges and adds them to the graph.

    Attributes
    ----------
    source_name : str
        The name of the source nodes.
    target_name : str
        The name of the target nodes.
    num_nearest_neighbours : int
        Number of nearest neighbours.

    Methods
    -------
    get_adjacency_matrix(source_nodes, target_nodes)
        Compute the adjacency matrix for the KNN method.
    register_edges(graph)
        Register the edges in the graph.
    register_attributes(graph, config)
        Register attributes in the edges of the graph.
    update_graph(graph, attrs_config)
        Update the graph with the edges.
    """

    def __init__(self, source_name: str, target_name: str, num_nearest_neighbours: int):
        super().__init__(source_name, target_name)
        assert isinstance(num_nearest_neighbours, int), "Number of nearest neighbours must be an integer"
        assert num_nearest_neighbours > 0, "Number of nearest neighbours must be positive"
        self.num_nearest_neighbours = num_nearest_neighbours

    def get_adjacency_matrix(self, source_nodes: np.ndarray, target_nodes: np.ndarray):
        """Compute the adjacency matrix for the KNN method.

        Parameters
        ----------
        source_nodes : np.ndarray
            The source nodes.
        target_nodes : np.ndarray
            The target nodes.
        """
        assert self.num_nearest_neighbours is not None, "number of neighbors required for knn encoder"
        LOGGER.info(
            "Using KNN-Edges (with %d nearest neighbours) between %s and %s.",
            self.num_nearest_neighbours,
            self.source_name,
            self.target_name,
        )

        nearest_neighbour = NearestNeighbors(metric="haversine", n_jobs=4)
        nearest_neighbour.fit(source_nodes.x.numpy())
        adj_matrix = nearest_neighbour.kneighbors_graph(
            target_nodes.x.numpy(),
            n_neighbors=self.num_nearest_neighbours,
            mode="distance",
        ).tocoo()
        return adj_matrix


class CutOffEdges(BaseEdgeBuilder):
    """Computes cut-off based edges and adds them to the graph.

    Attributes
    ----------
    source_name : str
        The name of the source nodes.
    target_name : str
        The name of the target nodes.
    cutoff_factor : float
        Factor to multiply the grid reference distance to get the cut-off radius.
    radius : float
        Cut-off radius.

    Methods
    -------
    get_cutoff_radius(graph, mask_attr)
        Compute the cut-off radius.
    get_adjacency_matrix(source_nodes, target_nodes)
        Get the adjacency matrix for the cut-off method.
    register_edges(graph)
        Register the edges in the graph.
    register_attributes(graph, config)
        Register attributes in the edges of the graph.
    update_graph(graph, attrs_config)
        Update the graph with the edges.
    """

    def __init__(self, source_name: str, target_name: str, cutoff_factor: float):
        super().__init__(source_name, target_name)
        assert isinstance(cutoff_factor, (int, float)), "Cutoff factor must be a float"
        assert cutoff_factor > 0, "Cutoff factor must be positive"
        self.cutoff_factor = cutoff_factor

    def get_cutoff_radius(self, graph: HeteroData, mask_attr: Optional[torch.Tensor] = None):
        """Compute the cut-off radius.

        The cut-off radius is computed as the product of the target nodes reference distance and the cut-off factor.

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
        mask = target_nodes[mask_attr] if mask_attr is not None else None
        target_grid_reference_distance = get_grid_reference_distance(target_nodes.x, mask)
        radius = target_grid_reference_distance * self.cutoff_factor
        return radius

    def prepare_node_data(self, graph: HeteroData) -> tuple[NodeStorage, NodeStorage]:
        """Prepare nodes information."""
        self.radius = self.get_cutoff_radius(graph)
        return super().prepare_node_data(graph)

    def get_adjacency_matrix(self, source_nodes: NodeStorage, target_nodes: NodeStorage):
        """Get the adjacency matrix for the cut-off method.

        Parameters
        ----------
        source_nodes : NodeStorage
            The source nodes.
        target_nodes : NodeStorage
            The target nodes.
        """
        LOGGER.info(
            "Using CutOff-Edges (with radius = %.1f km) between %s and %s.",
            self.radius * EARTH_RADIUS,
            self.source_name,
            self.target_name,
        )

        nearest_neighbour = NearestNeighbors(metric="haversine", n_jobs=4)
        nearest_neighbour.fit(source_nodes.x)
        adj_matrix = nearest_neighbour.radius_neighbors_graph(target_nodes.x, radius=self.radius).tocoo()
        return adj_matrix
