# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
"""Helpers for detecting fused multi-dataset graphs."""

from __future__ import annotations

from collections.abc import Mapping

from omegaconf import DictConfig
from torch_geometric.data import HeteroData

DEFAULT_DATASET_NAME = "data"
DEFAULT_EDGE_RELATION_NAME = "to"
DEFAULT_EDGE_WEIGHT_ATTRIBUTE = "gauss_weight"
DEFAULT_GAUSSIAN_NORM = "l1"


def get_graph_node_names(
    graph_or_config: HeteroData | DictConfig | Mapping,
) -> set[str]:
    """Return the node-type names visible in a built graph or graph config."""
    if isinstance(graph_or_config, HeteroData):
        return set(graph_or_config.node_types)

    if isinstance(graph_or_config, Mapping):
        nodes = graph_or_config.get("nodes", {})
    else:
        nodes = getattr(graph_or_config, "nodes", {})

    return set(nodes.keys()) if nodes else set()


def uses_fused_dataset_graph(graph_or_config: HeteroData | DictConfig | Mapping, dataset_names: list[str]) -> bool:
    """Return whether the graph has one node group per dataset.

    In this form each dataset name is itself a node group in the graph,
    rather than reusing a single generic ``data`` node group.
    """
    if not dataset_names:
        return False
    node_names = get_graph_node_names(graph_or_config)
    if not set(dataset_names).issubset(node_names):
        return False

    return dataset_names != [DEFAULT_DATASET_NAME] or DEFAULT_DATASET_NAME not in node_names


def participant_node_name(node_name: str, participant: str | None) -> str:
    """Return the node-group name of ``node_name`` for ``participant``."""
    return node_name if participant is None else f"{node_name}_{participant}"


def graph_participants(graph: HeteroData, node_name: str) -> list[str]:
    """Return the participants of a fused participant graph, in graph order.

    A graph that still carries the plain ``node_name`` node group is not a multi-participant
    graph and yields an empty list; otherwise the participants are the suffixes of the
    ``<node_name>_<participant>`` node groups produced by :func:`fuse_participant_graphs`.
    """
    node_types = list(graph.node_types)
    if node_name in node_types:
        return []

    prefix = f"{node_name}_"
    return [name[len(prefix) :] for name in node_types if name.startswith(prefix)]


def fuse_participant_graphs(graphs: Mapping[str, HeteroData]) -> HeteroData:
    """Fuse per-participant graphs into one graph with participant-suffixed node groups.

    Every node group of each graph is suffixed with its participant name
    (``data`` -> ``data_west``), together with the edge types connecting them. A single
    participant keeps the plain node-group names, so single-participant configurations
    produce exactly the graph they produce today.
    """
    if len(graphs) == 1:
        return next(iter(graphs.values()))

    fused = HeteroData()
    for participant, graph in graphs.items():
        for node_name, nodes in graph.node_items():
            fused_name = participant_node_name(node_name, participant)
            for key, value in nodes.items():
                fused[fused_name][key] = value
            fused[fused_name].num_nodes = nodes.num_nodes

        for (source_name, relation, target_name), edges in graph.edge_items():
            fused_edge = (
                participant_node_name(source_name, participant),
                relation,
                participant_node_name(target_name, participant),
            )
            for key, value in edges.items():
                fused[fused_edge][key] = value

    return fused
