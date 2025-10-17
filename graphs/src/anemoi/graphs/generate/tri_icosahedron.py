# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections.abc import Iterable

import networkx as nx
import numpy as np
import trimesh
from sklearn.neighbors import BallTree

from anemoi.graphs.generate.masks import KNNAreaMaskBuilder
from anemoi.graphs.generate.transforms import cartesian_to_latlon_rad
from anemoi.graphs.generate.utils import get_coordinates_ordering


def create_tri_nodes(
    resolution: int, area_mask_builder: KNNAreaMaskBuilder | None = None
) -> tuple[nx.DiGraph, np.ndarray, list[int]]:
    """Creates a global mesh from a refined icosahedron.

    This method relies on the trimesh python library.

    Parameters
    ----------
    resolution : int
        Level of mesh resolution to consider.
    area_mask_builder : KNNAreaMaskBuilder
        KNNAreaMaskBuilder with the cloud of points to limit the mesh area, by default None.

    Returns
    -------
    graph : networkx.Graph
        The specified graph (only nodes) sorted by latitude and longitude.
    coords_rad : np.ndarray
        The node coordinates (not ordered) in radians.
    node_ordering : list[int]
        Order of the node coordinates to be sorted by latitude and longitude.
    """
    coords_rad = get_latlon_coords_icosphere(resolution)

    node_ordering = get_coordinates_ordering(coords_rad)

    if area_mask_builder is not None:
        area_mask = area_mask_builder.get_mask(coords_rad)
        node_ordering = node_ordering[area_mask[node_ordering]]

    # Creates the graph, with the nodes sorted by latitude and longitude.
    nx_graph = create_nx_graph_from_tri_coords(coords_rad, node_ordering)

    return nx_graph, coords_rad, list(node_ordering)


def create_stretched_tri_nodes(
    base_resolution: int,
    lam_resolution: int,
    area_mask_builder: KNNAreaMaskBuilder | None = None,
) -> tuple[nx.DiGraph, np.ndarray, list[int]]:
    """Creates a global mesh with 2 levels of resolution.

    The base resolution is used to define the nodes outside the Area Of Interest (AOI),
    while the lam_resolution is used to define the nodes inside the AOI.

    Parameters
    ---------
    base_resolution : int
        Global resolution level.
    lam_resolution : int
        Local resolution level.
    area_mask_builder : KNNAreaMaskBuilder
        NearestNeighbors with the cloud of points to limit the mesh area.

    Returns
    -------
    nx_graph : nx.DiGraph
        The graph with the added nodes.
    coords_rad : np.ndarray
        The node coordinates (not ordered) in radians.
    node_ordering : list[int]
        Order of the node coordinates to be sorted by latitude and longitude.
    """
    assert area_mask_builder is not None, "AOI mask builder must be provided to build refined grid."
    # Get the low resolution nodes outside the AOI
    base_coords_rad = get_latlon_coords_icosphere(base_resolution)
    base_area_mask = ~area_mask_builder.get_mask(base_coords_rad)

    # Get the high resolution nodes inside the AOI
    lam_coords_rad = get_latlon_coords_icosphere(lam_resolution)
    lam_area_mask = area_mask_builder.get_mask(lam_coords_rad)

    coords_rad = np.concatenate([base_coords_rad[base_area_mask], lam_coords_rad[lam_area_mask]])

    node_ordering = get_coordinates_ordering(coords_rad)

    # Creates the graph, with the nodes sorted by latitude and longitude.
    nx_graph = create_nx_graph_from_tri_coords(coords_rad, node_ordering)

    return nx_graph, coords_rad, list(node_ordering)


def get_latlon_coords_icosphere(resolution: int) -> np.ndarray:
    """Get the latitude and longitude coordinates (in radians) of the icosphere.

    Parameters
    ----------
    resolution : int
        The resolution level of the icosphere.

    Returns
    -------
    np.ndarray
        The latitude and longitude coordinates, in radians, of the icosphere.
    """
    sphere = trimesh.creation.icosphere(subdivisions=resolution, radius=1.0)
    coords_rad = cartesian_to_latlon_rad(sphere.vertices)
    return coords_rad


def create_nx_graph_from_tri_coords(coords_rad: np.ndarray, node_ordering: np.ndarray) -> nx.DiGraph:
    """Creates the networkx graph from the coordinates and the node ordering."""
    graph = nx.DiGraph()
    for i, coords in enumerate(coords_rad[node_ordering]):
        node_id = node_ordering[i]
        graph.add_node(node_id, hcoords_rad=coords)

    assert list(graph.nodes.keys()) == list(node_ordering), "Nodes are not correctly added to the graph."
    assert graph.number_of_nodes() == len(node_ordering), "The number of nodes must be the same."
    return graph


def add_edges_to_nx_graph(
    graph: nx.DiGraph,
    resolutions: list[int],
    x_hops: int = 1,
    area_mask_builder: KNNAreaMaskBuilder | None = None,
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
    area_mask_builder : KNNAreaMaskBuilder
        NearestNeighbors with the cloud of points to limit the mesh area, by default None.

    Returns
    -------
    graph : nx.DiGraph
        The graph with the added edges.
    """
    assert x_hops > 0, "x_hops == 0, graph would have no edges ..."

    graph_vertices = np.array([graph.nodes[i]["hcoords_rad"] for i in sorted(graph.nodes)])
    tree = BallTree(graph_vertices, metric="haversine")

    # Build the multi-scale connections
    for resolution in resolutions:
        # Define the coordinates of the isophere vertices at specified 'resolution' level
        r_sphere = trimesh.creation.icosphere(subdivisions=resolution, radius=1.0)
        r_vertices_rad = cartesian_to_latlon_rad(r_sphere.vertices)

        # Limit area of mesh points.
        if area_mask_builder is not None:
            area_mask = area_mask_builder.get_mask(r_vertices_rad)
            valid_nodes = np.where(area_mask)[0]
        else:
            valid_nodes = None

        node_neighbours = get_neighbours_within_hops(r_sphere, x_hops, valid_nodes=valid_nodes)

        _, vertex_mapping_index = tree.query(r_vertices_rad, k=1)
        neighbour_pairs = create_node_neighbours_list(graph, node_neighbours, vertex_mapping_index)
        graph.add_edges_from(neighbour_pairs)
    return graph


def get_neighbours_within_hops(
    tri_mesh: trimesh.Trimesh, x_hops: int, valid_nodes: list[int] | None = None
) -> dict[int, set[int]]:
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

    if valid_nodes is not None:
        edges = edges[np.isin(tri_mesh.edges_unique, valid_nodes).all(axis=1)]
    else:
        valid_nodes = list(range(len(tri_mesh.vertices)))
    graph = nx.from_edgelist(edges)

    neighbours = {
        i: set(nx.ego_graph(graph, i, radius=x_hops, center=False) if i in graph else []) for i in valid_nodes
    }

    return neighbours


def add_neigbours_edges(
    graph: nx.Graph,
    node_idx: int,
    neighbour_indices: Iterable[int],
    self_loops: bool = False,
    vertex_mapping_index: np.ndarray | None = None,
) -> nx.Graph:
    """Adds the edges of one node to its neighbours.

    Parameters
    ----------
    graph : nx.Graph
        The graph.
    node_idx : int
        The node considered.
    neighbour_indices : list[int]
        The neighbours of the node.
    self_loops : bool, optional
        Whether is supported to add self-loops, by default False.
    vertex_mapping_index : np.ndarray, optional
        Index to map the vertices from the refined sphere to the original one, by default None.

    Returns
    -------
    nx.Graph
        The graph with the added edges.
    """
    graph_nodes_idx = list(sorted(graph.nodes))
    for neighbour_idx in neighbour_indices:
        if not self_loops and node_idx == neighbour_idx:  # no self-loops
            continue

        if vertex_mapping_index is not None:
            # Use the same method to add edge in all spheres
            node_neighbour = graph_nodes_idx[vertex_mapping_index[neighbour_idx][0]]
            node = graph_nodes_idx[vertex_mapping_index[node_idx][0]]
        else:
            node_neighbour = graph_nodes_idx[neighbour_idx]
            node = graph_nodes_idx[node_idx]

        # add edge to the graph
        if node in graph and node_neighbour in graph:
            graph.add_edge(node_neighbour, node)

    return graph


def create_node_neighbours_list(
    graph: nx.Graph,
    node_neighbours: dict[int, set[int]],
    vertex_mapping_index: np.ndarray | None = None,
    self_loops: bool = False,
) -> list[tuple]:
    """Preprocesses the dict of node neighbours.

    Parameters:
    -----------
    graph: nx.Graph
        The graph.
    node_neighbours: dict[int, set[int]]
        dictionairy with key: node index and value: set of neighbour node indices
    vertex_mapping_index: np.ndarry
        Index to map the vertices from the refined sphere to the original one, by default None.
    self_loops: bool
        Whether is supported to add self-loops, by default False.

    Returns:
    --------
    list: tuple
        A list with containing node neighbour pairs in tuples
    """
    graph_nodes_idx = list(sorted(graph.nodes))

    if vertex_mapping_index is None:
        vertex_mapping_index = np.arange(len(graph.nodes)).reshape(len(graph.nodes), 1)

    neighbour_pairs = [
        (graph_nodes_idx[vertex_mapping_index[node_neighbour][0]], graph_nodes_idx[vertex_mapping_index[node][0]])
        for node, neighbours in node_neighbours.items()
        for node_neighbour in neighbours
        if node != node_neighbour or (self_loops and node == node_neighbour)
    ]

    return neighbour_pairs
