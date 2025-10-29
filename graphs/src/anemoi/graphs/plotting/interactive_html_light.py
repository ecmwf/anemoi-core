from pathlib import Path

import numpy as np
import torch
from jinja2 import Template

HTML_TEMPLATE_PATH = Path(__file__).parent / "interactive.html.jinja"


def load_graph(
    path: str,
    nodes: list[str] = ["data", "hidden"],
    edges: list[str] = ["data_to_hidden", "hidden_to_hidden", "hidden_to_data"],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Load a hetero graph from a file and separate nodes and edges by type.

    Parameters
    ----------
    path : str
        Path to the graph file.
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
    hetero_data = torch.load(path, weights_only=False, map_location="cpu")
    out_nodes = {n: hetero_data[n].x for n in nodes}

    out_edges = {}
    for e in edges:
        # needed because in hierarchical graphs nodes names contain underscores...
        edge_key = e.split("_to_")
        edge_key = (edge_key[0], "to", edge_key[1])
        # -----------------
        out_edges[e] = hetero_data[edge_key].edge_index

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


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Generate HTML visualization of a graph.")
    parser.add_argument("graph_path", type=str, help="Path to the graph file.")
    parser.add_argument("output_path", type=str, help="Path to save the output HTML file.")
    parser.add_argument(
        "--nodes",
        type=lambda s: s.split(","),
        default="data,hidden",
        help="Comma-separated list of node types to extract.",
    )
    parser.add_argument(
        "--edges",
        type=lambda s: s.split(","),
        default="data_to_hidden,hidden_to_hidden,hidden_to_data",
        help="Comma-separated list of edge types to extract.",
    )
    args = parser.parse_args()

    print("Starting graph HTML generation...")
    print(f"Arguments: {args}")

    nodes, edges = load_graph(args.graph_path, nodes=args.nodes, edges=args.edges)

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
        nodes_embed.append({"name": node_set, "points": pts, "color": colors[i % len(colors)], "radius": 21 + i * 2})

    edges_embed = []
    for i, (edge_set, eds) in enumerate(edges.items()):
        edges_embed.append({"name": edge_set, "edges": eds})

    # # Render and save
    print("Rendering HTML...")
    with open(HTML_TEMPLATE_PATH, "r") as f:
        HTML_TEMPLATE = f.read()

    template = Template(HTML_TEMPLATE)
    html_output = template.render(nodes=nodes_embed, edges=edges_embed, max_degree=50, min_degree=1)

    with open(args.output_path, "w") as f:
        f.write(html_output)

    print(f"HTML file generated: {args.output_path}")
