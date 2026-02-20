# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import Any

from omegaconf import open_dict


def merge_projection_and_graph_config(graph_config: Any) -> None:
    """Merge projection nodes/edges into the main graph config in-place."""
    projections = getattr(graph_config, "projections", None) or {}
    if not projections:
        return

    with open_dict(graph_config):
        if graph_config.get("nodes") is None:
            graph_config.nodes = {}
        if graph_config.get("edges") is None:
            graph_config.edges = []

        for projection_cfg in projections.values():
            if projection_cfg is None:
                continue
            projection_nodes = projection_cfg.get("nodes", None)
            if projection_nodes:
                graph_config.nodes = {**graph_config.nodes, **projection_nodes}

            projection_edges = projection_cfg.get("edges", None)
            if projection_edges:
                graph_config.edges = list(graph_config.edges) + list(projection_edges)
