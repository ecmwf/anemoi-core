# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Helpers for building small projection subgraphs from compact configs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.graphs.projection_helpers import DEFAULT_EDGE_WEIGHT_ATTRIBUTE
from anemoi.graphs.projection_helpers import DEFAULT_GAUSSIAN_NORM


def _as_plain_mapping(cfg: Mapping | Any) -> dict[str, Any]:
    if OmegaConf.is_config(cfg):
        return dict(OmegaConf.to_container(cfg, resolve=True))
    return dict(cfg)


def _data_subgraph(graph_data: HeteroData, data_node_name: str) -> HeteroData:
    subgraph = HeteroData()
    subgraph[data_node_name].x = graph_data[data_node_name].x
    subgraph[data_node_name].num_nodes = graph_data[data_node_name].num_nodes
    return subgraph


def _update_graph(subgraph: HeteroData, config: Mapping[str, Any]) -> HeteroData:
    from anemoi.graphs.create import GraphCreator

    return GraphCreator(OmegaConf.create(config)).update_graph(subgraph)


def _knn_edge_cfg(
    source_name: str,
    target_name: str,
    num_nearest_neighbours: int,
    sigma: float,
) -> dict:
    return {
        "source_name": source_name,
        "target_name": target_name,
        "edge_builders": [
            {
                "_target_": "anemoi.graphs.edges.KNNEdges",
                "num_nearest_neighbours": num_nearest_neighbours,
            }
        ],
        "attributes": {
            DEFAULT_EDGE_WEIGHT_ATTRIBUTE: {
                "_target_": "anemoi.graphs.edges.attributes.GaussianDistanceWeights",
                "norm": DEFAULT_GAUSSIAN_NORM,
                "sigma": sigma,
            }
        },
    }


def _node_builder_config(
    config: Mapping[str, Any],
    *,
    grid_keys: tuple[str, ...],
    missing_message: str,
) -> Any:
    grid = next((config[key] for key in grid_keys if config.get(key) is not None), None)
    node_builder_cfg = config.get("node_builder")
    if node_builder_cfg is None:
        if grid is None:
            raise ValueError(missing_message)
        node_builder_cfg = {"_target_": "anemoi.graphs.nodes.ReducedGaussianGridNodes", "grid": grid}
    return node_builder_cfg


def build_graph_config_subgraph(
    graph_data: HeteroData,
    data_node_name: str,
    graph_config: Mapping | Any,
) -> HeteroData:
    """Build a subgraph from a graph config snippet."""
    graph_config = _as_plain_mapping(graph_config)
    nodes_cfg = graph_config.get("nodes")
    edges_cfg = graph_config.get("edges")
    if nodes_cfg is None or edges_cfg is None:
        msg = "Graph config subgraphs require both 'nodes' and 'edges'."
        raise ValueError(msg)
    return _update_graph(_data_subgraph(graph_data, data_node_name), {"nodes": nodes_cfg, "edges": edges_cfg})


def build_node_to_node_projection_subgraph(
    graph_data: HeteroData,
    source_node_name: str,
    target_node_name: str,
    target_node_config: Mapping | Any | None = None,
    *,
    target_grid_keys: tuple[str, ...] = ("grid",),
    num_nearest_neighbours: int | None = None,
    sigma: float | None = None,
    reverse: bool = False,
) -> HeteroData:
    """Build KNN projection edges between two node sets."""
    target_node_config = _as_plain_mapping(target_node_config) if target_node_config is not None else {}
    num_nearest_neighbours = (
        target_node_config.get("num_nearest_neighbours", 3)
        if num_nearest_neighbours is None
        else num_nearest_neighbours
    )
    sigma = target_node_config.get("sigma", 1.0) if sigma is None else sigma

    nodes = {}
    needs_target_node = target_node_name != source_node_name or target_node_config.get("node_builder") is not None
    needs_target_node = needs_target_node or any(target_node_config.get(key) is not None for key in target_grid_keys)
    if needs_target_node:
        node_builder_cfg = _node_builder_config(
            target_node_config,
            grid_keys=target_grid_keys,
            missing_message=f"Config for node '{target_node_name}' must specify a grid or node_builder.",
        )
        nodes[target_node_name] = {"node_builder": node_builder_cfg}

    edges = [_knn_edge_cfg(source_node_name, target_node_name, num_nearest_neighbours, sigma)]
    if reverse:
        edges.append(_knn_edge_cfg(target_node_name, source_node_name, num_nearest_neighbours, sigma))
    return _update_graph(
        _data_subgraph(graph_data, source_node_name),
        {"nodes": nodes, "edges": edges},
    )


def _expand_smoother_config(cfg: dict | Any) -> dict[str, dict]:
    """Return an explicit ``{name: spec}`` smoothers dict from cfg.

    Accepts either an already-explicit ``smoothers`` mapping or a compact
    geometric-progression spec (``num_scales``, ``base_num_nearest_neighbours``,
    ``base_sigma``, …).
    """
    if OmegaConf.is_config(cfg):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    smoothers = cfg.get("smoothers")
    if smoothers:
        return dict(smoothers)

    num_scales = cfg.get("num_scales")
    if num_scales is None:
        return {}

    base_neighbours = cfg["base_num_nearest_neighbours"]
    base_sigma = cfg["base_sigma"]
    scale_factor = cfg.get("scale_factor", 2)

    smoothers = {}
    for i in range(num_scales):
        factor = scale_factor**i
        smoothers[f"smooth_{factor}x"] = {
            "num_nearest_neighbours": base_neighbours * factor,
            "sigma": round(base_sigma * factor, 5),
        }
    return smoothers
