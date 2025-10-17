# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import h3
import networkx as nx
import numpy as np

from anemoi.graphs.generate.masks import KNNAreaMaskBuilder
from anemoi.graphs.generate.utils import get_coordinates_ordering


def create_hex_nodes(
    resolution: int,
    area_mask_builder: KNNAreaMaskBuilder | None = None,
) -> tuple[nx.Graph, np.ndarray, list[int]]:
    """Creates a global mesh from a refined icosahedron.

    This method relies on the H3 python library, which covers the earth with hexagons (and 5 pentagons). At each
    refinement level, a hexagon cell (nodes) has 7 child cells (aperture 7).

    Parameters
    ----------
    resolution : int
        Level of mesh resolution to consider.
    area_mask_builder : KNNAreaMaskBuilder, optional
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
    nodes = get_nodes_at_resolution(resolution)

    coords_rad = np.deg2rad(np.array([h3.h3_to_geo(node) for node in nodes]))

    node_ordering = get_coordinates_ordering(coords_rad)

    if area_mask_builder is not None:
        area_mask = area_mask_builder.get_mask(coords_rad)
        node_ordering = node_ordering[area_mask[node_ordering]]

    graph = create_nx_graph_from_hex_coords(nodes, node_ordering)

    return graph, coords_rad, list(node_ordering)


def create_nx_graph_from_hex_coords(nodes: list[str], node_ordering: np.ndarray) -> nx.Graph:
    """Add all nodes at a specified refinement level to a graph.

    Parameters
    ----------
    nodes : list[str]
        The set of H3 indexes (nodes).
    node_ordering: np.ndarray
        Order of the node coordinates to be sorted by latitude and longitude.

    Returns
    -------
    graph : networkx.Graph
        The graph with the added nodes.
    """
    graph = nx.Graph()

    for node_pos in node_ordering:
        h3_idx = nodes[node_pos]
        graph.add_node(h3_idx, hcoords_rad=np.deg2rad(h3.h3_to_geo(h3_idx)))

    return graph


def get_nodes_at_resolution(
    resolution: int,
) -> list[str]:
    """Get nodes at a specified refinement level over the entire globe.

    Parameters
    ----------
    resolution : int
        The H3 refinement level. It can be an integer from 0 to 15.

    Returns
    -------
    nodes : list[str]
        The list of H3 indexes at the specified resolution level.
    """
    return list(h3.uncompact(h3.get_res0_indexes(), resolution))


def add_edges_to_nx_graph(
    graph: nx.Graph,
    resolutions: list[int],
    x_hops: int = 1,
    depth_children: int = 0,
) -> nx.Graph:
    """Adds the edges to the graph.

    This method includes multi-scale connections to the existing graph. The different scales
    are defined by the resolutions (or refinement levels) specified.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to add the edges.
    resolutions : list[int]
        Levels of mesh resolution to consider.
    x_hops: int
        The number of hops to consider for the neighbours.
    depth_children : int
        The number of resolution levels to consider for the connections of children. Defaults to 1, which includes
        connections up to the next resolution level.

    Returns
    -------
    graph : networkx.Graph
        The graph with the added edges.
    """

    graph = add_neighbour_edges(graph, resolutions, x_hops)
    graph = add_edges_to_children(graph, resolutions, depth_children)
    return graph


def add_neighbour_edges(
    graph: nx.Graph,
    refinement_levels: tuple[int],
    x_hops: int = 1,
) -> nx.Graph:
    """Adds edges between neighbours at the specified refinement levels."""
    for resolution in refinement_levels:
        nodes = select_nodes_from_graph_at_resolution(graph, resolution)

        for idx in nodes:
            # neighbours
            for idx_neighbour in h3.k_ring(idx, k=x_hops) & set(nodes):
                graph = add_edge(
                    graph,
                    h3.h3_to_center_child(idx, max(refinement_levels)),
                    h3.h3_to_center_child(idx_neighbour, max(refinement_levels)),
                )

    return graph


def add_edges_to_children(
    graph: nx.Graph,
    refinement_levels: tuple[int],
    depth_children: int | None = None,
) -> nx.Graph:
    """Adds edges to the children of the nodes at the specified resolution levels.

    Parameters
    ----------
    graph : nx.Graph
        graph to which the edges will be added
    refinement_levels : tuple[int]
        set of refinement levels
    depth_children : int, optional
        The number of resolution levels to consider for the connections of children. Defaults to 1, which includes
        connections up to the next resolution level, by default None.

    Returns
    -------
    nx.Graph
        Graph with the added edges.
    """
    if depth_children is None:
        depth_children = len(refinement_levels)
    elif depth_children == 0:
        return graph

    for i_level, resolution_parent in enumerate(list(sorted(refinement_levels))[0:-1]):
        parent_nodes = select_nodes_from_graph_at_resolution(graph, resolution_parent)

        for parent_idx in parent_nodes:
            # add own children
            for resolution_child in refinement_levels[i_level + 1 : i_level + depth_children + 1]:
                for child_idx in h3.h3_to_children(parent_idx, res=resolution_child):
                    graph = add_edge(
                        graph,
                        h3.h3_to_center_child(parent_idx, refinement_levels[-1]),
                        h3.h3_to_center_child(child_idx, refinement_levels[-1]),
                    )

    return graph


def select_nodes_from_graph_at_resolution(graph: nx.Graph, resolution: int) -> set[str]:
    """Select nodes from a graph at a specified resolution level."""
    nodes_at_lower_resolution = [n for n in h3.compact(graph.nodes) if h3.h3_get_resolution(n) <= resolution]
    nodes_at_resolution = h3.uncompact(nodes_at_lower_resolution, resolution)
    return nodes_at_resolution


def add_edge(
    graph: nx.Graph,
    source_node_h3_idx: str,
    target_node_h3_idx: str,
) -> nx.Graph:
    """Add edge between two nodes to a graph.

    The edge will only be added in case both target and source nodes are included in the graph.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to add the nodes.
    source_node_h3_idx : str
        The H3 index of the tail of the edge.
    target_node_h3_idx : str
        The H3 index of the head of the edge.

    Returns
    -------
    graph : networkx.Graph
        The graph with the added edge.
    """
    if not graph.has_node(source_node_h3_idx) or not graph.has_node(target_node_h3_idx):
        return graph

    if source_node_h3_idx != target_node_h3_idx:
        graph.add_edge(source_node_h3_idx, target_node_h3_idx)

    return graph
