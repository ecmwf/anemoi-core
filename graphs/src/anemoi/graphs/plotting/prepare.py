# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import torch
from torch_geometric.data import HeteroData


def node_list(graph: HeteroData, nodes_name: str, mask: list[bool] | None = None) -> tuple[list[float], list[float]]:
    """Get the latitude and longitude of the nodes.

    Parameters
    ----------
    graph : dict[str, torch.Tensor]
        Graph to plot.
    nodes_name : str
        Name of the nodes.
    mask : list[bool], optional
        Mask to filter the nodes. Default is None.

    Returns
    -------
    latitudes : list[float]
        Latitude coordinates of the nodes.
    longitudes : list[float]
        Longitude coordinates of the nodes.
    """
    coords = np.rad2deg(graph[nodes_name].x.numpy())
    latitudes = coords[:, 0]
    longitudes = coords[:, 1]
    if mask is not None:
        latitudes = latitudes[mask]
        longitudes = longitudes[mask]
    return latitudes.tolist(), longitudes.tolist()


def edge_list(graph: HeteroData, source_nodes_name: str, target_nodes_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Get the edge list.

    This method returns the edge list to be represented in a graph. It computes the coordinates of the points connected
    and include NaNs to separate the edges.

    Parameters
    ----------
    graph : HeteroData
        Graph to plot.
    source_nodes_name : str
        Name of the source nodes.
    target_nodes_name : str
        Name of the target nodes.

    Returns
    -------
    latitudes : np.ndarray
        Latitude coordinates of the points connected.
    longitudes : np.ndarray
        Longitude coordinates of the points connected.
    """
    sub_graph = graph[(source_nodes_name, "to", target_nodes_name)].edge_index
    x0 = np.rad2deg(graph[source_nodes_name].x[sub_graph[0]])
    y0 = np.rad2deg(graph[target_nodes_name].x[sub_graph[1]])
    nans = np.full_like(x0[:, :1], np.nan)
    latitudes = np.concatenate([x0[:, :1], y0[:, :1], nans], axis=1).flatten()
    longitudes = np.concatenate([x0[:, 1:2], y0[:, 1:2], nans], axis=1).flatten()
    return latitudes, longitudes


def compute_node_adjacencies(
    graph: HeteroData, source_nodes_name: str, target_nodes_name: str
) -> tuple[list[int], list[str]]:
    """Compute the number of adjacencies of each target node in a bipartite graph.

    Parameters
    ----------
    graph : HeteroData
        Graph to plot.
    source_nodes_name : str
        Name of the dimension of the coordinates for the head nodes.
    target_nodes_name : str
        Name of the dimension of the coordinates for the tail nodes.

    Returns
    -------
    num_adjacencies : np.ndarray
        Number of adjacencies of each node.
    """
    node_adjacencies = np.zeros(graph[target_nodes_name].num_nodes, dtype=int)
    vals, counts = np.unique(graph[(source_nodes_name, "to", target_nodes_name)].edge_index[1], return_counts=True)
    node_adjacencies[vals] = counts
    return node_adjacencies


def get_node_adjancency_attributes(graph: HeteroData) -> dict[str, tuple[str, np.ndarray]]:
    """Get the node adjacencies for each subgraph."""
    node_adj_attr = {}
    for (source_nodes_name, _, target_nodes_name), _ in graph.edge_items():
        attr_name = f"# connections from {source_nodes_name}"
        node_adj_vector = compute_node_adjacencies(graph, source_nodes_name, target_nodes_name)
        if target_nodes_name not in node_adj_attr:
            node_adj_attr[target_nodes_name] = {attr_name: node_adj_vector}
        else:
            node_adj_attr[target_nodes_name][attr_name] = node_adj_vector

    return node_adj_attr


def compute_isolated_nodes(graph: HeteroData) -> dict[str, tuple[list, list]]:
    """Compute the isolated nodes.

    Parameters
    ----------
    graph : HeteroData
        Graph.

    Returns
    -------
    dict[str, list[int]]
        Dictionary with the isolated nodes for each subgraph.
    """
    isolated_nodes = {}
    for (source_name, _, target_name), sub_graph in graph.edge_items():
        head_isolated = np.ones(graph[source_name].num_nodes, dtype=bool)
        tail_isolated = np.ones(graph[target_name].num_nodes, dtype=bool)
        head_isolated[sub_graph.edge_index[0]] = False
        tail_isolated[sub_graph.edge_index[1]] = False
        if np.any(head_isolated):
            isolated_nodes[f"{source_name} isolated (--> {target_name})"] = node_list(
                graph, source_name, mask=list(head_isolated)
            )
        if np.any(tail_isolated):
            isolated_nodes[f"{target_name} isolated ({source_name} -->)"] = node_list(
                graph, target_name, mask=list(tail_isolated)
            )

    return isolated_nodes


def get_node_attribute_dims(graph: HeteroData) -> dict[str, int]:
    """Get dimensions of the node attributes.

    Parameters
    ----------
    graph : HeteroData
        The graph to inspect.

    Returns
    -------
    dict[str, int]
        A dictionary with the attribute names as keys and the number of dimensions as values.
    """
    attr_dims = {}
    for nodes in graph.node_stores:
        for attr in nodes.node_attrs():
            if attr == "x" or not isinstance(nodes[attr], torch.Tensor):
                continue
            elif attr not in attr_dims:
                attr_dims[attr] = nodes[attr].shape[1]
            else:
                assert (
                    nodes[attr].shape[1] == attr_dims[attr]
                ), f"Attribute {attr} has different dimensions in different nodes."
    return attr_dims


def get_edge_attribute_dims(graph: HeteroData) -> dict[str, int]:
    """Get dimensions of the node attributes.

    Parameters
    ----------
    graph : HeteroData
        The graph to inspect.

    Returns
    -------
    dict[str, int]
        A dictionary with the attribute names as keys and the number of dimensions as values.
    """
    attr_dims = {}
    for edges in graph.edge_stores:
        for attr in edges.edge_attrs():
            if attr == "edge_index":
                continue
            elif attr not in attr_dims:
                attr_dims[attr] = edges[attr].shape[1]
            else:
                assert (
                    edges[attr].shape[1] == attr_dims[attr]
                ), f"Attribute {attr} has different dimensions in different edges."
    return attr_dims
