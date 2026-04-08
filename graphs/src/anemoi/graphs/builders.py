# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Helpers for building small projection subgraphs from compact configs.

These are used by model/loss components (TruncatedConnection,
MultiscaleLossWrapper) that own their own subgraphs rather than relying on
projection edges merged into the main graph.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.graphs.projection_helpers import DEFAULT_EDGE_WEIGHT_ATTRIBUTE


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
    edge_weight_attribute = cfg.get("edge_weight_attribute", DEFAULT_EDGE_WEIGHT_ATTRIBUTE)
    gaussian_norm = cfg.get("gaussian_norm", "l1")

    smoothers = {}
    for i in range(num_scales):
        factor = scale_factor**i
        smoothers[f"smooth_{factor}x"] = {
            "edge_weight_attribute": edge_weight_attribute,
            "gaussian_norm": gaussian_norm,
            "num_nearest_neighbours": base_neighbours * factor,
            "sigma": round(base_sigma * factor, 5),
        }
    return smoothers


def _knn_edge_cfg(
    source_name: str,
    target_name: str,
    num_nearest_neighbours: int,
    edge_weight_attribute: str,
    gaussian_norm: str,
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
            edge_weight_attribute: {
                "_target_": "anemoi.graphs.edges.attributes.GaussianDistanceWeights",
                "norm": gaussian_norm,
                "sigma": sigma,
            }
        },
    }


def build_truncation_subgraph(
    graph_data: HeteroData,
    data_node_name: str,
    truncation_config: Mapping | Any,
) -> HeteroData:
    """Build a ``HeteroData`` with data→truncation and truncation→data edges.

    The returned subgraph contains the data nodes (copied from *graph_data*)
    and truncation nodes built from the grid/node_builder spec in
    *truncation_config*.  Edge names are:

    - ``(data_node_name, "to", "truncation")`` — down-projection
    - ``("truncation", "to", data_node_name)`` — up-projection

    Parameters
    ----------
    graph_data:
        Main graph; data-node coordinates are read from here.
    data_node_name:
        Node type in *graph_data* that corresponds to the data grid.
    truncation_config:
        Mapping with keys: ``grid`` (or ``node_builder``),
        ``num_nearest_neighbours``, ``edge_weight_attribute``, ``sigma``,
        ``gaussian_norm``.
    """
    from anemoi.graphs.create import GraphCreator

    if OmegaConf.is_config(truncation_config):
        truncation_config = OmegaConf.to_container(truncation_config, resolve=True)

    grid = truncation_config.get("grid") or truncation_config.get("truncation_grid")
    node_builder_cfg = truncation_config.get("node_builder")
    if node_builder_cfg is None:
        if grid is None:
            msg = "truncation_config must specify 'grid' or 'node_builder'."
            raise ValueError(msg)
        node_builder_cfg = {"_target_": "anemoi.graphs.nodes.ReducedGaussianGridNodes", "grid": grid}

    num_nearest_neighbours = truncation_config.get("num_nearest_neighbours", 3)
    edge_weight_attribute = truncation_config.get("edge_weight_attribute", DEFAULT_EDGE_WEIGHT_ATTRIBUTE)
    gaussian_norm = truncation_config.get("gaussian_norm", "l1")
    sigma = truncation_config.get("sigma", 1.0)

    subgraph = HeteroData()
    subgraph[data_node_name].x = graph_data[data_node_name].x
    subgraph[data_node_name].num_nodes = graph_data[data_node_name].num_nodes

    config = OmegaConf.create(
        {
            "nodes": {"truncation": {"node_builder": node_builder_cfg}},
            "edges": [
                _knn_edge_cfg(data_node_name, "truncation", num_nearest_neighbours, edge_weight_attribute, gaussian_norm, sigma),
                _knn_edge_cfg("truncation", data_node_name, num_nearest_neighbours, edge_weight_attribute, gaussian_norm, sigma),
            ],
        }
    )

    return GraphCreator(config).update_graph(subgraph)


def build_smoother_subgraph(
    graph_data: HeteroData,
    data_node_name: str,
    smoother_config: Mapping | Any,
) -> HeteroData:
    """Build a ``HeteroData`` with self-loop smoother edges over data nodes.

    The subgraph contains only the data nodes (copied from *graph_data*) with
    KNN self-loop edges.  Edge name is
    ``(data_node_name, "to", data_node_name)``.

    Parameters
    ----------
    graph_data:
        Main graph; data-node coordinates are read from here.
    data_node_name:
        Node type in *graph_data* that corresponds to the data grid.
    smoother_config:
        Single-smoother mapping with keys: ``num_nearest_neighbours``,
        ``sigma``, ``edge_weight_attribute``, ``gaussian_norm``.
    """
    from anemoi.graphs.create import GraphCreator

    if OmegaConf.is_config(smoother_config):
        smoother_config = OmegaConf.to_container(smoother_config, resolve=True)

    num_nearest_neighbours = smoother_config["num_nearest_neighbours"]
    edge_weight_attribute = smoother_config.get("edge_weight_attribute", DEFAULT_EDGE_WEIGHT_ATTRIBUTE)
    gaussian_norm = smoother_config.get("gaussian_norm", "l1")
    sigma = smoother_config["sigma"]

    subgraph = HeteroData()
    subgraph[data_node_name].x = graph_data[data_node_name].x
    subgraph[data_node_name].num_nodes = graph_data[data_node_name].num_nodes

    config = OmegaConf.create(
        {
            "nodes": {},
            "edges": [
                _knn_edge_cfg(data_node_name, data_node_name, num_nearest_neighbours, edge_weight_attribute, gaussian_norm, sigma),
            ],
        }
    )

    return GraphCreator(config).update_graph(subgraph)
