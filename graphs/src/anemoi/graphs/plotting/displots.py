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
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import EdgeStorage
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.plotting.prepare import compute_node_adjacencies
from anemoi.graphs.plotting.prepare import get_edge_attribute_dims
from anemoi.graphs.plotting.prepare import get_node_attribute_dims

LOGGER = logging.getLogger(__name__)


def plot_distribution_node_attributes(graph: HeteroData, out_file: str | Path | None = None) -> None:
    """Figure with the distribution of the node attributes.

    Each row represents a node type and each column an attribute dimension.
    """
    num_nodes = len(graph.node_types)
    attr_dims = get_node_attribute_dims(graph)
    plot_distribution_attributes(graph.node_items(), num_nodes, attr_dims, "Node", out_file)


def plot_distribution_edge_attributes(graph: HeteroData, out_file: str | Path | None = None) -> None:
    """Figure with the distribution of the edge attributes.

    Each row represents a edge type and each column an attribute dimension.
    """
    num_edges = len(graph.edge_types)
    attr_dims = get_edge_attribute_dims(graph)
    plot_distribution_attributes(graph.edge_items(), num_edges, attr_dims, "Edge", out_file)


def plot_distribution_node_derived_attributes(graph, outfile: str | Path | None = None):
    """Figure with the distribution of the node derived attributes.

    Each row represents a node type and each column an attribute dimension.
    """
    node_adjs = {}
    node_attr_dims = {}
    for source_name, _, target_name in graph.edge_types:
        node_adj_tensor = compute_node_adjacencies(graph, source_name, target_name)
        node_adj_tensor = torch.from_numpy(node_adj_tensor.reshape((node_adj_tensor.shape[0], -1)))
        node_adj_key = f"# edges from {source_name}"

        # Store node adjacencies
        if target_name in node_adjs:
            node_adjs[target_name] = node_adjs[target_name] | {node_adj_key: node_adj_tensor}
        else:
            node_adjs[target_name] = {node_adj_key: node_adj_tensor}

        # Store attribute dimension
        if node_adj_key not in node_attr_dims:
            node_attr_dims[node_adj_key] = node_adj_tensor.shape[1]

    node_adj_list = [(k, v) for k, v in node_adjs.items()]

    plot_distribution_attributes(node_adj_list, len(node_adjs), node_attr_dims, "Node", outfile)


def plot_distribution_attributes(
    graph_items: NodeStorage | EdgeStorage,
    num_items: int,
    attr_dims: dict,
    item_type: Literal["Edge", "Node"],
    out_file: str | Path | None = None,
) -> None:
    """Figure with the distribution of the node and edge attributes.

    Each row represents a node or edge type and each column an attribute dimension.
    """
    dim_attrs = sum(attr_dims.values())

    if dim_attrs == 0:
        LOGGER.warning("No edge attributes found in the graph.")
        return None

    # Define the layout
    _, axs = plt.subplots(num_items, dim_attrs, figsize=(10 * num_items, 10))
    if num_items == dim_attrs == 1:
        axs = np.array([[axs]])
    elif axs.ndim == 1:
        axs = axs.reshape(num_items, dim_attrs)

    for i, (item_name, item_store) in enumerate(graph_items):
        for j, (attr_name, attr_values) in enumerate(attr_dims.items()):
            for dim in range(attr_values):
                if attr_name in item_store:
                    axs[i, j + dim].hist(item_store[attr_name][:, dim].float(), bins=50)

                    axs[i, j + dim].set_ylabel("".join(item_name).replace("to", " --> "))
                    axs[i, j + dim].set_title(attr_name if attr_values == 1 else f"{attr_name}_{dim}")
                    if i == num_items - 1:
                        axs[i, j + dim].set_xlabel(attr_name if attr_values == 1 else f"{attr_name}_{dim}")
                else:
                    axs[i, j + dim].set_axis_off()

    plt.suptitle(f"{item_type} Attributes distribution", fontsize=14)
    plt.savefig(out_file)
