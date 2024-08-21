from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import numpy as np
import trimesh
from sklearn.metrics.pairwise import haversine_distances
from sklearn.neighbors import BallTree

from anemoi.graphs.generate.transforms import cartesian_to_latlon_rad


def create_icosahedral_nodes(
    resolutions: list[int],
) -> tuple[nx.DiGraph, np.ndarray, list[int]]:
    """Creates a global mesh following AIFS strategy.

    This method relies on the trimesh python library.

    Parameters
    ----------
    resolutions : list[int]
        Levels of mesh resolution to consider.
    aoi_mask_builder : KNNAreaMaskBuilder
        KNNAreaMaskBuilder with the cloud of points to limit the mesh area, by default None.

    Returns
    -------
    graph : networkx.Graph
        The specified graph (nodes & edges).
    coords_rad : np.ndarray
        The node coordinates (not ordered) in radians.
    node_ordering : list[int]
        Order of the nodes in the graph to be sorted by latitude and longitude.
    """
    sphere = trimesh.creation.icosphere(subdivisions=resolutions[-1], radius=1.0)

    coords_rad = cartesian_to_latlon_rad(sphere.vertices)

    node_ordering = get_node_ordering(coords_rad)

    # TODO: AOI mask builder is not used in the current implementation.

    nx_graph = create_icosahedral_nx_graph_from_coords(coords_rad, node_ordering)

    return nx_graph, coords_rad, list(node_ordering)


def create_icosahedral_nx_graph_from_coords(coords_rad: np.ndarray, node_ordering: list[int]):

    graph = nx.DiGraph()
    for i, coords in enumerate(coords_rad[node_ordering]):
        node_id = node_ordering[i]
        graph.add_node(node_id, hcoords_rad=coords)

    assert list(graph.nodes.keys()) == list(node_ordering), "Nodes are not correctly added to the graph."
    assert graph.number_of_nodes() == len(node_ordering), "The number of nodes must be the same."
    return graph


def get_node_ordering(coords_rad: np.ndarray) -> np.ndarray:
    """Get the node ordering to sort the nodes by latitude and longitude."""
    # Get indices to sort points by lon & lat in radians.
    index_latitude = np.argsort(coords_rad[:, 1])
    index_longitude = np.argsort(coords_rad[index_latitude][:, 0])[::-1]
    node_ordering = np.arange(coords_rad.shape[0])[index_latitude][index_longitude]
    return node_ordering


def add_edges_to_nx_graph(
    graph: nx.DiGraph,
    resolutions: list[int],
    x_hops: int = 1,
) -> nx.DiGraph:
    """Adds the edges to the graph.

    This method adds multi-scale connections to the existing graph. The corresponfing nodes or vertices
    are defined by an isophere at the different esolutions (or refinement levels) specified.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to add the edges. It should correspond to the mesh nodes, without edges.
    resolutions : list[int]
        Levels of mesh refinement levels to consider.
    x_hops : int, optional
        Number of hops between 2 nodes to consider them neighbours, by default 1.
    aoi_mask_builder : KNNAreaMaskBuilder
        NearestNeighbors with the cloud of points to limit the mesh area, by default None.
    margin_radius_km : float, optional
        Margin radius in km to consider when creating the processor mesh, by default 0.0.

    Returns
    -------
    graph : nx.DiGraph
        The graph with the added edges.
    """
    assert x_hops > 0, "x_hops == 0, graph would have no edges ..."

    sphere = trimesh.creation.icosphere(subdivisions=resolutions[-1], radius=1.0)
    vertices_rad = cartesian_to_latlon_rad(sphere.vertices)
    node_neighbours = get_neighbours_within_hops(sphere, x_hops)

    for idx_node, idx_neighbours in node_neighbours.items():
        add_neigbours_edges(graph, vertices_rad, idx_node, idx_neighbours)

    tree = BallTree(vertices_rad, metric="haversine")

    # Build the multi-scale connections
    for resolution in resolutions[:-1]:
        # Define the isophere at specified 'resolution' level
        r_sphere = trimesh.creation.icosphere(subdivisions=resolution, radius=1.0)

        # Get the vertices of the isophere
        r_vertices_rad = cartesian_to_latlon_rad(r_sphere.vertices)

        # TODO AOI mask builder is not used in the current implementation.

        node_neighbours = get_neighbours_within_hops(r_sphere, x_hops)

        _, vertex_mapping_index = tree.query(r_vertices_rad, k=1)
        for idx_node, idx_neighbours in node_neighbours.items():
            add_neigbours_edges(
                graph, r_vertices_rad, idx_node, idx_neighbours, vertex_mapping_index=vertex_mapping_index
            )

    return graph


def get_neighbours_within_hops(tri_mesh: trimesh.Trimesh, x_hops: int) -> dict[int, set[int]]:
    """Get the neigbour connections in the graph.

    Parameters
    ----------
    tri_mesh : trimesh.Trimesh
        The mesh to consider.
    x_hops : int
        Number of hops between 2 nodes to consider them neighbours.
    valid_nodes : list[int], optional
        List of valid nodes to consider, by default None. It is useful to consider only a subset of the nodes to save
        computation time.

    Returns
    -------
    neighbours : dict[int, set[int]]
        A list with the neighbours for each vertex. The element at position 'i' correspond to the neighbours to the
        i-th vertex of the mesh.
    """
    edges = tri_mesh.edges_unique

    valid_nodes = list(range(len(tri_mesh.vertices)))
    graph = nx.from_edgelist(edges)

    # Get a dictionary of the neighbours within 'x_hops' neighbourhood of each node in the graph
    neighbours = {
        i: set(nx.ego_graph(graph, i, radius=x_hops, center=False) if i in graph else []) for i in valid_nodes
    }

    return neighbours


def add_neigbours_edges(
    graph: nx.Graph,
    vertices: np.ndarray,
    node_idx: int,
    neighbour_indices: Iterable[int],
    self_loops: bool = False,
    vertex_mapping_index: np.ndarray | None = None,
) -> None:
    """Adds the edges of one node to its neighbours.

    Parameters
    ----------
    graph : nx.Graph
        The graph.
    vertices : np.ndarray
        A 2D array of shape (num_vertices, 2) with the planar coordinates of the mesh, in radians.
    node_idx : int
        The node considered.
    neighbours : list[int]
        The neighbours of the node.
    self_loops : bool, optional
        Whether is supported to add self-loops, by default False.
    vertex_mapping_index : np.ndarray, optional
        Index to map the vertices from the refined sphere to the original one, by default None.
    """
    for neighbour_idx in neighbour_indices:
        if not self_loops and node_idx == neighbour_idx:  # no self-loops
            continue

        location_node = vertices[node_idx]
        location_neighbour = vertices[neighbour_idx]
        edge_length = haversine_distances([location_neighbour, location_node])[0][1]

        if vertex_mapping_index is not None:
            # Use the same method to add edge in all spheres
            node_neighbour = vertex_mapping_index[neighbour_idx][0]
            node = vertex_mapping_index[node_idx][0]
        else:
            node, node_neighbour = node_idx, neighbour_idx

        # add edge to the graph (if both source and target nodes are in the graph)
        if node in graph and node_neighbour in graph:
            graph.add_edge(node_neighbour, node, weight=edge_length)
