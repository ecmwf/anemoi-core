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
from abc import ABC
from abc import abstractmethod

import networkx as nx
import numpy as np
import scipy
import torch
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs import EARTH_RADIUS
from anemoi.graphs.generate import hex_icosahedron
from anemoi.graphs.generate import tri_icosahedron
from anemoi.graphs.generate.masks import KNNAreaMaskBuilder
from anemoi.graphs.nodes.builders.from_refined_icosahedron import HexNodes
from anemoi.graphs.nodes.builders.from_refined_icosahedron import LimitedAreaHexNodes
from anemoi.graphs.nodes.builders.from_refined_icosahedron import LimitedAreaTriNodes
from anemoi.graphs.nodes.builders.from_refined_icosahedron import StretchedTriNodes
from anemoi.graphs.nodes.builders.from_refined_icosahedron import TriNodes
from anemoi.graphs.utils import concat_edges
from anemoi.graphs.utils import get_grid_reference_distance

LOGGER = logging.getLogger(__name__)


class BaseEdgeBuilder(ABC):
    """Base class for edge builders."""

    def __init__(
        self,
        source_name: str,
        target_name: str,
        source_mask_attr_name: str | None = None,
        target_mask_attr_name: str | None = None,
    ):
        self.source_name = source_name
        self.target_name = target_name
        self.source_mask_attr_name = source_mask_attr_name
        self.target_mask_attr_name = target_mask_attr_name

    @property
    def name(self) -> tuple[str, str, str]:
        """Name of the edge subgraph."""
        return self.source_name, "to", self.target_name

    @abstractmethod
    def get_adjacency_matrix(self, source_nodes: NodeStorage, target_nodes: NodeStorage): ...

    def prepare_node_data(self, graph: HeteroData) -> tuple[NodeStorage, NodeStorage]:
        """Prepare node information and get source and target nodes."""
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
        edge_index = self.get_edge_index(graph)
        edge_type = type(self).__name__

        if "edge_index" in graph[self.name]:
            # Expand current edge indices
            graph[self.name].edge_index = concat_edges(graph[self.name].edge_index, edge_index)
            if edge_type not in graph[self.name].edge_type:
                graph[self.name].edge_type = graph[self.name].edge_type + "," + edge_type
            return graph

        # Register new edge indices
        graph[self.name].edge_index = edge_index
        graph[self.name].edge_type = edge_type
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

    def update_graph(self, graph: HeteroData, attrs_config: DotDict | None = None) -> HeteroData:
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

        if attrs_config is not None:
            graph = self.register_attributes(graph, attrs_config)

        return graph


class NodeMaskingMixin:
    """Mixin class for masking source/target nodes when building edges."""

    def get_node_coordinates(
        self, source_nodes: NodeStorage, target_nodes: NodeStorage
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the node coordinates."""
        source_coords, target_coords = source_nodes.x.numpy(), target_nodes.x.numpy()

        if self.source_mask_attr_name is not None:
            source_coords = source_coords[source_nodes[self.source_mask_attr_name].squeeze()]

        if self.target_mask_attr_name is not None:
            target_coords = target_coords[target_nodes[self.target_mask_attr_name].squeeze()]

        return source_coords, target_coords

    def undo_masking(self, adj_matrix, source_nodes: NodeStorage, target_nodes: NodeStorage):
        if self.target_mask_attr_name is not None:
            target_mask = target_nodes[self.target_mask_attr_name].squeeze()
            assert adj_matrix.shape[0] == target_mask.sum()
            target_mapper = dict(zip(list(range(adj_matrix.shape[0])), np.where(target_mask)[0]))
            adj_matrix.row = np.vectorize(target_mapper.get)(adj_matrix.row)

        if self.source_mask_attr_name is not None:
            source_mask = source_nodes[self.source_mask_attr_name].squeeze()
            assert adj_matrix.shape[1] == source_mask.sum()
            source_mapper = dict(zip(list(range(adj_matrix.shape[1])), np.where(source_mask)[0]))
            adj_matrix.col = np.vectorize(source_mapper.get)(adj_matrix.col)

        if self.source_mask_attr_name is not None or self.target_mask_attr_name is not None:
            true_shape = target_nodes.x.shape[0], source_nodes.x.shape[0]
            adj_matrix = coo_matrix((adj_matrix.data, (adj_matrix.row, adj_matrix.col)), shape=true_shape)

        return adj_matrix


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
        assert isinstance(num_nearest_neighbours, int), "Number of nearest neighbours must be an integer"
        assert num_nearest_neighbours > 0, "Number of nearest neighbours must be positive"
        self.num_nearest_neighbours = num_nearest_neighbours

    def get_adjacency_matrix(self, source_nodes: NodeStorage, target_nodes: NodeStorage) -> np.ndarray:
        """Compute the adjacency matrix for the KNN method.

        Parameters
        ----------
        source_nodes : NodeStorage
            The source nodes.
        target_nodes : NodeStorage
            The target nodes.

        Returns
        -------
        np.ndarray
            The adjacency matrix.
        """
        source_coords, target_coords = self.get_node_coordinates(source_nodes, target_nodes)
        assert self.num_nearest_neighbours is not None, "number of neighbors required for knn encoder"
        LOGGER.info(
            "Using KNN-Edges (with %d nearest neighbours) between %s and %s.",
            self.num_nearest_neighbours,
            self.source_name,
            self.target_name,
        )

        nearest_neighbour = NearestNeighbors(metric="haversine", n_jobs=4)
        nearest_neighbour.fit(source_coords)
        adj_matrix = nearest_neighbour.kneighbors_graph(
            target_coords,
            n_neighbors=self.num_nearest_neighbours,
            mode="distance",
        ).tocoo()

        # Post-process the adjacency matrix. Add masked nodes.
        adj_matrix = self.undo_masking(adj_matrix, source_nodes, target_nodes)

        return adj_matrix


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
    ):
        super().__init__(source_name, target_name, source_mask_attr_name, target_mask_attr_name)
        assert isinstance(cutoff_factor, (int, float)), "Cutoff factor must be a float"
        assert cutoff_factor > 0, "Cutoff factor must be positive"
        self.cutoff_factor = cutoff_factor

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
        mask = target_nodes[mask_attr] if mask_attr is not None else None
        target_grid_reference_distance = get_grid_reference_distance(target_nodes.x, mask)
        radius = target_grid_reference_distance * self.cutoff_factor
        return radius

    def prepare_node_data(self, graph: HeteroData) -> tuple[NodeStorage, NodeStorage]:
        """Prepare node information and get source and target nodes."""
        self.radius = self.get_cutoff_radius(graph)
        return super().prepare_node_data(graph)

    def get_adjacency_matrix(self, source_nodes: NodeStorage, target_nodes: NodeStorage) -> np.ndarray:
        """Get the adjacency matrix for the cut-off method.

        Parameters
        ----------
        source_nodes : NodeStorage
            The source nodes.
        target_nodes : NodeStorage
            The target nodes.

        Returns
        -------
        np.ndarray
            The adjacency matrix.
        """
        source_coords, target_coords = self.get_node_coordinates(source_nodes, target_nodes)
        LOGGER.info(
            "Using CutOff-Edges (with radius = %.1f km) between %s and %s.",
            self.radius * EARTH_RADIUS,
            self.source_name,
            self.target_name,
        )

        nearest_neighbour = NearestNeighbors(metric="haversine", n_jobs=4)
        nearest_neighbour.fit(source_coords)
        adj_matrix = nearest_neighbour.radius_neighbors_graph(target_coords, radius=self.radius).tocoo()

        # Post-process the adjacency matrix. Add masked nodes.
        adj_matrix = self.undo_masking(adj_matrix, source_nodes, target_nodes)

        return adj_matrix


class MultiScaleEdges(BaseEdgeBuilder):
    """Base class for multi-scale edges in the nodes of a graph.

    Attributes
    ----------
    source_name : str
        The name of the source nodes.
    target_name : str
        The name of the target nodes.
    x_hops : int
        Number of hops (in the refined icosahedron) between two nodes to connect
        them with an edge.

    Methods
    -------
    register_edges(graph)
        Register the edges in the graph.
    register_attributes(graph, config)
        Register attributes in the edges of the graph.
    update_graph(graph, attrs_config)
        Update the graph with the edges.
    """

    VALID_NODES = [
        TriNodes,
        HexNodes,
        LimitedAreaTriNodes,
        LimitedAreaHexNodes,
        StretchedTriNodes,
    ]

    def __init__(self, source_name: str, target_name: str, x_hops: int, **kwargs):
        super().__init__(source_name, target_name)
        assert source_name == target_name, f"{self.__class__.__name__} requires source and target nodes to be the same."
        assert isinstance(x_hops, int), "Number of x_hops must be an integer"
        assert x_hops > 0, "Number of x_hops must be positive"
        self.x_hops = x_hops

    def add_edges_from_tri_nodes(self, nodes: NodeStorage) -> NodeStorage:
        nodes["_nx_graph"] = tri_icosahedron.add_edges_to_nx_graph(
            nodes["_nx_graph"],
            resolutions=nodes["_resolutions"],
            x_hops=self.x_hops,
            area_mask_builder=nodes.get("_area_mask_builder", None),
        )

        return nodes

    def add_edges_from_stretched_tri_nodes(self, nodes: NodeStorage) -> NodeStorage:
        all_points_mask_builder = KNNAreaMaskBuilder("all_nodes", 1.0)
        all_points_mask_builder.fit_coords(nodes.x.numpy())

        nodes["_nx_graph"] = tri_icosahedron.add_edges_to_nx_graph(
            nodes["_nx_graph"],
            resolutions=nodes["_resolutions"],
            x_hops=self.x_hops,
            area_mask_builder=all_points_mask_builder,
        )
        return nodes

    def add_edges_from_hex_nodes(self, nodes: NodeStorage) -> NodeStorage:
        nodes["_nx_graph"] = hex_icosahedron.add_edges_to_nx_graph(
            nodes["_nx_graph"],
            resolutions=nodes["_resolutions"],
            x_hops=self.x_hops,
        )

        return nodes

    def get_adjacency_matrix(self, source_nodes: NodeStorage, target_nodes: NodeStorage):
        if source_nodes.node_type in [TriNodes.__name__, LimitedAreaTriNodes.__name__]:
            source_nodes = self.add_edges_from_tri_nodes(source_nodes)
        elif source_nodes.node_type in [HexNodes.__name__, LimitedAreaHexNodes.__name__]:
            source_nodes = self.add_edges_from_hex_nodes(source_nodes)
        elif source_nodes.node_type == StretchedTriNodes.__name__:
            source_nodes = self.add_edges_from_stretched_tri_nodes(source_nodes)
        else:
            raise ValueError(f"Invalid node type {source_nodes.node_type}")

        adjmat = nx.to_scipy_sparse_array(source_nodes["_nx_graph"], format="coo")

        return adjmat

    def update_graph(self, graph: HeteroData, attrs_config: DotDict | None = None) -> HeteroData:
        node_type = graph[self.source_name].node_type
        valid_node_names = [n.__name__ for n in self.VALID_NODES]
        assert node_type in valid_node_names, f"{self.__class__.__name__} requires {','.join(valid_node_names)} nodes."

        return super().update_graph(graph, attrs_config)


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
