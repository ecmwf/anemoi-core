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

    return set(nodes.keys())


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
