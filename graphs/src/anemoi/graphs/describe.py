# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math
from itertools import chain
from pathlib import Path

import torch

from anemoi.utils.humanize import bytes
from anemoi.utils.text import table


class GraphDescriptor:
    """Class for descripting the graph."""

    def __init__(self, path: str | Path, **kwargs):
        self.path = path
        self.graph = torch.load(self.path, weights_only=False, map_location="cpu")

    @property
    def total_size(self):
        """Total size of the tensors in the graph (in bytes)."""
        total_size = 0

        for store in chain(self.graph.node_stores, self.graph.edge_stores):
            for value in store.values():
                if isinstance(value, torch.Tensor):
                    total_size += value.numel() * value.element_size()

        return total_size

    def get_node_summary(self) -> list[list]:
        """Summary of the nodes in the graph.

        Returns
        -------
        list[list]
            Returns a list for each subgraph with the following information:
            - Node name.
            - Number of nodes.
            - List of attribute names.
            - Total dimension of the attributes.
            - Min. latitude.
            - Max. latitude.
            - Min. longitude.
            - Max. longitude.
        """
        node_summary = []
        for name, nodes in self.graph.node_items():
            attributes = nodes.node_attrs()
            attributes.remove("x")

            node_summary.append(
                [
                    name,
                    nodes.num_nodes,
                    ", ".join(attributes),
                    sum(nodes[attr].shape[1] for attr in attributes if isinstance(nodes[attr], torch.Tensor)),
                    nodes.x[:, 0].min().item() / 2 / math.pi * 360,
                    nodes.x[:, 0].max().item() / 2 / math.pi * 360,
                    nodes.x[:, 1].min().item() / 2 / math.pi * 360,
                    nodes.x[:, 1].max().item() / 2 / math.pi * 360,
                ]
            )
        return node_summary

    def get_edge_summary(self) -> list[list]:
        """Summary of the edges in the graph.

        Returns
        -------
        list[list]
            Returns a list for each subgraph with the following information:
            - Source node name.
            - Destination node name.
            - Number of edges.
            - Number of isolated source nodes.
            - Number of isolated target nodes.
            - Min edges per target node.
            - Max edges per target node.
            - Total dimension of the attributes.
            - List of attribute names.
        """
        edge_summary = []
        for (src_name, _, dst_name), edges in self.graph.edge_items():
            attributes = edges.edge_attrs()
            attributes.remove("edge_index")

            # Compute edge counts per target node
            target_edge_counts = torch.bincount(edges.edge_index[1], minlength=self.graph[dst_name].num_nodes)
            # Only consider nodes that have at least one edge
            connected_target_counts = target_edge_counts[target_edge_counts > 0]
            min_edges_per_target = connected_target_counts.min().item() if len(connected_target_counts) > 0 else 0
            max_edges_per_target = connected_target_counts.max().item() if len(connected_target_counts) > 0 else 0

            edge_summary.append(
                [
                    src_name,
                    dst_name,
                    edges.num_edges,
                    self.graph[src_name].num_nodes - len(torch.unique(edges.edge_index[0])),
                    self.graph[dst_name].num_nodes - len(torch.unique(edges.edge_index[1])),
                    min_edges_per_target,
                    max_edges_per_target,
                    sum(edges[attr].shape[1] for attr in attributes),
                    ", ".join([f"{attr}({edges[attr].shape[1]}D)" for attr in attributes]),
                ]
            )
        return edge_summary

    def get_node_attribute_table(self) -> list[list]:
        node_attributes = []
        for node_name, node_store in self.graph.node_items():
            node_attr_names = node_store.node_attrs()
            node_attr_names.remove("x")  # Remove the coordinates from statistics table
            for node_attr_name in node_attr_names:
                node_attributes.append(
                    [
                        "Node",
                        node_name,
                        node_attr_name,
                        node_store[node_attr_name].dtype,
                        node_store[node_attr_name].float().min().item(),
                        node_store[node_attr_name].float().mean().item(),
                        node_store[node_attr_name].float().max().item(),
                        node_store[node_attr_name].float().std().item(),
                    ]
                )
        return node_attributes

    def get_edge_attribute_table(self) -> list[list]:
        edge_attributes = []
        for (source_name, _, target_name), edge_store in self.graph.edge_items():
            edge_attr_names = edge_store.edge_attrs()
            edge_attr_names.remove("edge_index")  # Remove the edge index from statistics table
            for edge_attr_name in edge_attr_names:
                edge_attributes.append(
                    [
                        "Edge",
                        f"{source_name}-->{target_name}",
                        edge_attr_name,
                        edge_store[edge_attr_name].dtype,
                        edge_store[edge_attr_name].float().min().item(),
                        edge_store[edge_attr_name].float().mean().item(),
                        edge_store[edge_attr_name].float().max().item(),
                        edge_store[edge_attr_name].float().std().item(),
                    ]
                )

        return edge_attributes

    def get_attribute_table(self) -> list[list]:
        """Get a table with the attributes of the graph."""
        attribute_table = []
        attribute_table.extend(self.get_node_attribute_table())
        attribute_table.extend(self.get_edge_attribute_table())
        return attribute_table

    def describe(self, show_attribute_distributions: bool = True) -> None:
        """Describe the graph."""
        print()
        print(f"📦 Path       : {self.path}")
        print(f"💽 Size       : {bytes(self.total_size)} ({self.total_size})")
        print()
        print("🪩  Nodes summary")
        print()
        print(
            table(
                self.get_node_summary(),
                header=[
                    "Nodes name",
                    "Num. nodes",
                    "Attributes",
                    "Attribute dim",
                    "Min. latitude",
                    "Max. latitude",
                    "Min. longitude",
                    "Max. longitude",
                ],
                align=["<", ">", ">", ">", ">", ">", ">", ">"],
                margin=3,
            )
        )
        print()
        print()
        print("🌐  Edges summary")
        print()
        print(
            table(
                self.get_edge_summary(),
                header=[
                    "Source",
                    "Target",
                    "Num. edges",
                    "Isolated Source",
                    "Isolated Target",
                    "Min edges/target",
                    "Max edges/target",
                    "Attribute dim",
                    "Attributes",
                ],
                align=["<", "<", ">", ">", ">", ">", ">", ">", ">"],
                margin=3,
            )
        )
        print()
        if show_attribute_distributions:
            print()
            print("📊 Attribute distributions")
            print()
            print(
                table(
                    self.get_attribute_table(),
                    header=[
                        "Type",
                        "Source",
                        "Name",
                        "Dtype",
                        "Min.",
                        "Mean",
                        "Max.",
                        "Std. dev.",
                    ],
                    align=["<", "<", ">", ">", ">", ">", ">", ">"],
                    margin=3,
                )
            )
            print()
        print("🔋 Graph ready.")
        print()
