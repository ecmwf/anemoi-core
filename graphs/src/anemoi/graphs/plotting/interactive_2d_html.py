# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from pathlib import Path
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from matplotlib.colors import rgb2hex
from torch_geometric.data import HeteroData

from anemoi.graphs.plotting.prepare import compute_isolated_nodes
from anemoi.graphs.plotting.prepare import compute_node_adjacencies
from anemoi.graphs.plotting.prepare import edge_list
from anemoi.graphs.plotting.prepare import node_list

annotations_style = {"text": "", "showarrow": False, "xref": "paper", "yref": "paper", "x": 0.005, "y": -0.002}
plotly_axis_config = {"showgrid": False, "zeroline": False, "showticklabels": False}

LOGGER = logging.getLogger(__name__)


def plot_interactive_subgraph_2d(
    graph: HeteroData,
    edges_to_plot: tuple[str, str, str],
    out_file: Optional[Union[str, Path]] = None,
) -> None:
    """Plots a bipartite graph (bi-graph).

    This methods plots the bipartite graph passed in an interactive window (using Ploty).

    Parameters
    ----------
    graph : dict
        The graph to plot.
    edges_to_plot : tuple[str, str]
        Names of the edges to plot.
    out_file : str | Path, optional
        Name of the file to save the plot. Default is None.
    """
    source_name, _, target_name = edges_to_plot
    edge_x, edge_y = edge_list(graph, source_nodes_name=source_name, target_nodes_name=target_name)
    assert source_name in graph.node_types, f"edges_to_plot ({source_name}) should be in the graph"
    assert target_name in graph.node_types, f"edges_to_plot ({target_name}) should be in the graph"
    lats_source_nodes, lons_source_nodes = node_list(graph, source_name)
    lats_target_nodes, lons_target_nodes = node_list(graph, target_name)

    # Compute node adjacencies
    node_adjacencies = compute_node_adjacencies(graph, source_name, target_name)
    node_text = [f"# of connections: {x}" for x in node_adjacencies]

    edge_trace = go.Scattergeo(
        lat=edge_x,
        lon=edge_y,
        line={"width": 0.5, "color": "#888"},
        hoverinfo="none",
        mode="lines",
        name="Connections",
    )

    source_node_trace = go.Scattergeo(
        lat=lats_source_nodes,
        lon=lons_source_nodes,
        mode="markers",
        hoverinfo="text",
        name=source_name,
        marker={
            "showscale": False,
            "color": "red",
            "size": 2,
            "line_width": 2,
        },
    )

    target_node_trace = go.Scattergeo(
        lat=lats_target_nodes,
        lon=lons_target_nodes,
        mode="markers",
        hoverinfo="text",
        name=target_name,
        text=node_text,
        marker={
            "showscale": True,
            "colorscale": "YlGnBu",
            "reversescale": True,
            "color": list(node_adjacencies),
            "size": 10,
            "colorbar": {"thickness": 15, "title": "Node Connections", "xanchor": "left", "title_side": "right"},
            "line_width": 2,
        },
    )
    layout = go.Layout(
        title="<br>" + f"Graph {source_name} --> {target_name}",
        title_font_size=16,
        showlegend=True,
        hovermode="closest",
        margin={"b": 20, "l": 5, "r": 5, "t": 40},
        annotations=[annotations_style],
        legend={"x": 0, "y": 1},
        xaxis=plotly_axis_config,
        yaxis=plotly_axis_config,
    )
    fig = go.Figure(data=[edge_trace, source_node_trace, target_node_trace], layout=layout)
    fig.update_geos(fitbounds="locations")

    if out_file is not None:
        fig.write_html(out_file)
    else:
        fig.show()


def plot_isolated_nodes_2d(graph: HeteroData, out_file: Optional[Union[str, Path]] = None) -> None:
    """Plot isolated nodes.

    This method creates an interactive visualization of the isolated nodes in the graph.

    Parameters
    ----------
    graph : AnemoiGraph
        The graph to plot.
    out_file : str | Path, optional
        Name of the file to save the plot. Default is None.
    """
    isolated_nodes = compute_isolated_nodes(graph)

    if len(isolated_nodes) == 0:
        LOGGER.warning("No isolated nodes found.")
        return

    colorbar = plt.cm.rainbow(np.linspace(0, 1, len(isolated_nodes)))
    nodes = []
    for name, (lat, lon) in isolated_nodes.items():
        nodes.append(
            go.Scattergeo(
                lat=lat,
                lon=lon,
                mode="markers",
                hoverinfo="text",
                name=name,
                marker={"showscale": False, "color": rgb2hex(colorbar[len(nodes)]), "size": 10},
            ),
        )

    layout = go.Layout(
        title="<br>Orphan nodes",
        title_font_size=16,
        showlegend=True,
        hovermode="closest",
        margin={"b": 20, "l": 5, "r": 5, "t": 40},
        annotations=[annotations_style],
        legend={"x": 0, "y": 1},
        xaxis=plotly_axis_config,
        yaxis=plotly_axis_config,
    )
    fig = go.Figure(data=nodes, layout=layout)
    fig.update_geos(fitbounds="locations")

    if out_file is not None:
        fig.write_html(out_file)
    else:
        fig.show()


def plot_interactive_nodes_2d(graph: HeteroData, nodes_name: str, out_file: Optional[str] = None) -> None:
    """Plot nodes.

    This method creates an interactive visualization of a set of nodes.

    Parameters
    ----------
    graph : HeteroData
        Graph.
    nodes_name : str
        Name of the nodes to plot.
    out_file : str, optional
        Name of the file to save the plot. Default is None.
    """
    node_latitudes, node_longitudes = node_list(graph, nodes_name)
    node_attrs = graph[nodes_name].node_attrs()
    # Remove x to avoid plotting the coordinates as an attribute
    node_attrs.remove("x")

    if len(node_attrs) == 0:
        LOGGER.warning(f"No node attributes found for {nodes_name} nodes.")
        return

    node_traces = {}
    for node_attr in node_attrs:
        node_attr_values = graph[nodes_name][node_attr].float().numpy()

        if node_attr_values.ndim == 1:
            node_attr_values = torch.unsqueeze(node_attr_values, -1)

        for attr_dim in range(node_attr_values.shape[1]):
            suffix = "" if node_attr_values.shape[1] == 1 else f"_[{attr_dim}]"
            node_traces[node_attr + suffix] = go.Scattergeo(
                lat=node_latitudes,
                lon=node_longitudes,
                name=" ".join((node_attr + suffix).split("_")).capitalize(),
                mode="markers",
                hoverinfo="text",
                marker={
                    "color": node_attr_values[:, attr_dim].squeeze().tolist(),
                    "showscale": True,
                    "colorscale": "RdBu",
                    "colorbar": {"thickness": 15, "title": node_attr + suffix, "xanchor": "left"},
                    "size": 5,
                },
                visible=False,
            )

    # Create and add slider
    slider_steps = []
    for i, node_attr in enumerate(node_traces.keys()):
        step = dict(
            label=f"Node attribute: {node_attr}",
            method="update",
            args=[{"visible": [False] * len(node_traces)}],
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        slider_steps.append(step)

    fig = go.Figure(
        data=list(node_traces.values()),
        layout=go.Layout(
            title=f"<br>Map of {nodes_name} nodes",
            sliders=[
                dict(active=0, currentvalue={"visible": False}, len=0.4, x=0.5, xanchor="center", steps=slider_steps)
            ],
            title_font_size=16,
            showlegend=False,
            hovermode="closest",
            margin={"b": 20, "l": 5, "r": 5, "t": 40},
            annotations=[annotations_style],
            xaxis=plotly_axis_config,
            yaxis=plotly_axis_config,
        ),
    )
    fig.data[0].visible = True

    if out_file is not None:
        fig.write_html(out_file)
    else:
        fig.show()
