# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path

import numpy as np
import torch
from jinja2 import Template
from torch_geometric.data import HeteroData

HTML_TEMPLATE_PATH = Path(__file__).parent / "interactive_3d.html.jinja"


def extract_nodes_edges(
    graph: HeteroData,
    nodes: list[str] | None = None,
    edges: list[str] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Extracts nodes and edges from a heterogeneous graph.

    Optionally filters by specified types.

    Parameters
    ----------
    graph : HeteroData
        The graph to subset.
    nodes : list[str]
        List of node types to extract.
    edges : list[str]
        List of edge types to extract.

    Returns
    -------
    tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
        A tuple containing two dictionaries:
        - The first dictionary maps node types to their coordinates tensors.
        - The second dictionary maps edge types to their edge index tensors.
    """
    if nodes:
        for n in nodes:
            if n not in graph.node_types:
                raise ValueError(f"Node type '{n}' not found in the graph.")
    else:
        nodes = graph.node_types

    if edges:
        edges_keys = ()
        for e in edges:
            edge_key = e.split("_to_")
            edge_key = (edge_key[0], "to", edge_key[1])
            if edge_key not in graph.edge_types:
                raise ValueError(f"Edge type '{e}' not found in the graph.")
            edges_keys += (edge_key,)
    else:
        edges_keys = graph.edge_types

    out_nodes = {n: graph[n].x for n in nodes}
    out_edges = {"_".join(e): graph[e].edge_index for e in edges_keys}

    return out_nodes, out_edges


def coords_to_latlon(coordinates):
    """Convert radians coordinates to latitude and longitude in degrees."""
    coordinates = np.rad2deg(coordinates)
    lats, lons = coordinates.T
    return lats, lons


def to_nodes_json(lats, lons, prefix="P"):
    """Convert nodes dictionary to JSON format for HTML rendering."""
    assert len(lats) == len(lons)
    names = [f"{prefix}_{i}" for i in range(len(lats))]
    points = [{"name": n, "pos": [float(lat), float(lon)]} for n, lat, lon in zip(names, lats, lons)]
    return points


def to_edges_json(names1, names2, pairs):
    edges = [[names1[i], names2[j]] for i, j in pairs]
    return edges


def plot_interactive_graph_3d(
    graph: HeteroData,
    out_file: str | Path,
) -> None:
    """Plot the entire graph in 3D.

    This method creates an interactive 3D visualization of the entire graph.

    Parameters
    ----------
    graph : dict
        The graph to plot.
    out_file : str | Path, optional
        Name of the file to save the plot. Default is None.
    """
    nodes, edges = extract_nodes_edges(graph)

    for node_set in nodes:
        node_lats, node_lons = coords_to_latlon(nodes[node_set].numpy())
        nodes[node_set] = to_nodes_json(node_lats, node_lons, prefix=node_set)

    for edge_set in edges:
        src_nodes, dst_nodes = edge_set.split("_to_")
        src_names = [f"{src_nodes}_{i}" for i in range(len(nodes[src_nodes]))]
        dst_names = [f"{dst_nodes}_{i}" for i in range(len(nodes[dst_nodes]))]
        edges[edge_set] = to_edges_json(src_names, dst_names, edges[edge_set].numpy().T)

    colors = [
        "#5050ff",
        "#ff5050",
        "#50ff50",
        "#ffaa00",
        "#aa00ff",
        "#00aaff",
        "#ff0055",
        "#55ff00",
        "#00ffaa",
        "#ff5500",
        "#0055ff",
        "#aa5500",
    ]

    nodes_embed = []
    for i, (node_set, pts) in enumerate(nodes.items()):
        nodes_embed.append({"name": node_set, "points": pts, "color": colors[i % len(colors)], "radius": 20.1 + i * 2})

    edges_embed = []
    for i, (edge_set, eds) in enumerate(edges.items()):
        edges_embed.append({"name": edge_set, "edges": eds})

    # # Render and save
    with open(HTML_TEMPLATE_PATH, "r") as f:
        HTML_TEMPLATE = f.read()

    template = Template(HTML_TEMPLATE)
    html_output = template.render(nodes=nodes_embed, edges=edges_embed, max_degree=50, min_degree=1)
    with open(out_file, "w") as f:
        f.write(html_output)
