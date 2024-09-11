from __future__ import annotations

import h3
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import haversine_distances


def create_hexagonal_nodes(
    resolutions: list[int],
    area: dict | None = None,
) -> tuple[nx.Graph, np.ndarray, list[int]]:
    """Creates a global mesh from a refined icosahedro.

    This method relies on the H3 python library, which covers the earth with hexagons (and 5 pentagons). At each
    refinement level, a hexagon cell (nodes) has 7 child cells (aperture 7).

    Parameters
    ----------
    resolutions : list[int]
        Levels of mesh resolution to consider.
    area : dict
        A region, in GeoJSON data format, to be contained by all cells. Defaults to None, which computes the global
        mesh.

    Returns
    -------
    graph : networkx.Graph
        The specified graph (nodes & edges).
    coords_rad : np.ndarray
        The node coordinates (not ordered) in radians.
    node_ordering : list[int]
        Order of the nodes in the graph to be sorted by latitude and longitude.
    """
    graph = nx.Graph()

    area_kwargs = {"area": area}

    for resolution in resolutions:
        graph = add_nodes_for_resolution(graph, resolution, **area_kwargs)

    coords = np.deg2rad(np.array([h3.h3_to_geo(node) for node in graph.nodes]))

    # Sort nodes by latitude and longitude
    node_ordering = np.lexsort(coords.T[::-1], axis=0)

    return graph, coords, list(node_ordering)


def add_nodes_for_resolution(
    graph: nx.Graph,
    resolution: int,
    **area_kwargs: dict | None,
) -> nx.Graph:
    """Add all nodes at a specified refinement level to a graph.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to add the nodes.
    resolution : int
        The H3 refinement level. It can be an integer from 0 to 15.
    area_kwargs: dict
        Additional arguments to pass to the get_nodes_at_resolution function.
    """

    nodes = get_nodes_at_resolution(resolution, **area_kwargs)

    for idx in nodes:
        graph.add_node(idx, hcoords_rad=np.deg2rad(h3.h3_to_geo(idx)))

    return graph


def get_nodes_at_resolution(
    resolution: int,
    area: dict | None = None,
) -> set[str]:
    """Get nodes at a specified refinement level over the entire globe.

    If area is not None, it will return the nodes within the specified area

    Parameters
    ----------
    resolution : int
        The H3 refinement level. It can be an integer from 0 to 15.
    area : dict
        An area as GeoJSON dictionary specifying a polygon. Defaults to None.

    Returns
    -------
    nodes : set[str]
        The set of H3 indexes at the specified resolution level.
    """
    nodes = h3.uncompact(h3.get_res0_indexes(), resolution) if area is None else h3.polyfill(area, resolution)

    # TODO: AOI not used in the current implementation.

    return nodes


def add_edges_to_nx_graph(
    graph: nx.Graph,
    resolutions: list[int],
    x_hops: int = 1,
    depth_children: int = 1,
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
    graph = add_edges_to_children(
        graph,
        resolutions,
        depth_children,
    )
    return graph


def add_neighbour_edges(
    graph: nx.Graph,
    refinement_levels: tuple[int],
    x_hops: int = 1,
) -> nx.Graph:
    for resolution in refinement_levels:
        nodes = select_nodes_from_graph_at_resolution(graph, resolution)

        for idx in nodes:
            # neighbours
            for idx_neighbour in h3.k_ring(idx, k=x_hops) & set(nodes):
                graph = add_edge(
                    graph,
                    h3.h3_to_center_child(idx, refinement_levels[-1]),
                    h3.h3_to_center_child(idx_neighbour, refinement_levels[-1]),
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
    depth_children : Optional[int], optional
        The number of resolution levels to consider for the connections of children. Defaults to 1, which includes
        connections up to the next resolution level, by default None
    """
    if depth_children is None:
        depth_children = len(refinement_levels)

    for i_level, resolution_parent in enumerate(refinement_levels[0:-1]):
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


def select_nodes_from_graph_at_resolution(graph: nx.Graph, resolution: int):
    parent_nodes = [node for node in graph.nodes if h3.h3_get_resolution(node) == resolution]
    return parent_nodes


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
    """
    if not graph.has_node(source_node_h3_idx) or not graph.has_node(target_node_h3_idx):
        return graph

    if source_node_h3_idx != target_node_h3_idx:
        source_location = np.deg2rad(h3.h3_to_geo(source_node_h3_idx))
        target_location = np.deg2rad(h3.h3_to_geo(target_node_h3_idx))
        graph.add_edge(
            source_node_h3_idx, target_node_h3_idx, weight=haversine_distances([source_location, target_location])[0][1]
        )

    return graph
