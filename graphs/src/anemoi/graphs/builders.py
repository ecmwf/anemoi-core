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

import logging
from collections.abc import Mapping
from typing import Any

from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.graphs.projection_helpers import DEFAULT_EDGE_WEIGHT_ATTRIBUTE
from anemoi.graphs.projection_helpers import DEFAULT_GAUSSIAN_NORM

LOGGER = logging.getLogger(__name__)


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


def build_node_to_node_projection_subgraph(
    graph_data: HeteroData,
    source_node_name: str,
    target_node_name: str,
    config: Mapping | Any,
) -> HeteroData:
    """Build a subgraph with KNN edges from source to target nodes.

    If ``source_node_name == target_node_name``, builds self-loop edges.
    If ``target_node_name`` is already in *graph_data*, its coordinates are
    copied by reference. Otherwise target nodes are built from
    ``config['node_builder']`` or ``config['grid']``.

    Parameters
    ----------
    graph_data:
        Main graph; source (and optionally target) node coordinates are read
        from here.
    source_node_name:
        Node type for the source (and KNN query) side.
    target_node_name:
        Node type for the target side.
    config:
        Mapping with keys: ``num_nearest_neighbours`` (int), ``sigma`` (float),
        and optionally ``node_builder`` or ``grid`` when the target nodes must
        be created rather than copied from *graph_data*.

    Returns
    -------
    HeteroData
        Subgraph with source (and target) node coordinates and KNN edges.
    """
    from anemoi.graphs.create import GraphCreator

    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)

    missing = [key for key in ("num_nearest_neighbours", "sigma") if config.get(key) is None]
    if missing:
        msg = (
            f"build_node_to_node_projection_subgraph requires config keys {missing}: "
            "'num_nearest_neighbours' (int) and 'sigma' (float) are mandatory."
        )
        raise ValueError(msg)

    num_nearest_neighbours = config["num_nearest_neighbours"]
    sigma = config["sigma"]

    subgraph = HeteroData()
    subgraph[source_node_name].x = graph_data[source_node_name].x
    subgraph[source_node_name].num_nodes = graph_data[source_node_name].num_nodes

    graph_config: dict = {
        "nodes": {},
        "edges": [_knn_edge_cfg(source_node_name, target_node_name, num_nearest_neighbours, sigma)],
    }

    if source_node_name == target_node_name:
        pass  # self-loop: target node data already set above
    elif target_node_name in graph_data.node_types:
        subgraph[target_node_name].x = graph_data[target_node_name].x
        subgraph[target_node_name].num_nodes = graph_data[target_node_name].num_nodes
    else:
        node_builder_cfg = config.get("node_builder")
        if node_builder_cfg is None:
            grid = config.get("grid")
            if grid is None:
                msg = (
                    f"Target node '{target_node_name}' not in graph_data; config must provide 'node_builder' or 'grid'."
                )
                raise ValueError(msg)
            node_builder_cfg = {"_target_": "anemoi.graphs.nodes.ReducedGaussianGridNodes", "grid": grid}
        graph_config["nodes"][target_node_name] = {"node_builder": node_builder_cfg}

    return GraphCreator(OmegaConf.create(graph_config)).update_graph(subgraph)


def build_truncation_subgraph(
    graph_data: HeteroData,
    data_node_name: str,
    truncation_config: Mapping | Any,
) -> HeteroData:
    """Build a new subgraph containing data nodes and a coarser truncation grid.

    To avoid copying data-node coordinates, this creates a fresh ``HeteroData``
    but shares data-node coordinates with *graph_data* by reference and adds
    KNN edges to a new ``truncation`` node group.

    Parameters
    ----------
    graph_data:
        Main graph; data-node coordinates are read from here by reference.
    data_node_name:
        Node type in *graph_data* that holds the data-grid coordinates.
    truncation_config:
        Compact mapping with keys:

        - ``grid`` or ``node_builder``: truncation-grid specification
        - ``num_nearest_neighbours``: int (default 3)
        - ``sigma``: Gaussian width (default 1.0)

    Returns
    -------
    HeteroData
        A new subgraph with ``data_node_name`` and ``truncation`` node types
        and KNN edges between them.
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
    sigma = truncation_config.get("sigma", 1.0)

    subgraph = HeteroData()
    subgraph[data_node_name].x = graph_data[data_node_name].x
    subgraph[data_node_name].num_nodes = graph_data[data_node_name].num_nodes

    config = OmegaConf.create(
        {
            "nodes": {"truncation": {"node_builder": node_builder_cfg}},
            "edges": [
                _knn_edge_cfg(data_node_name, "truncation", num_nearest_neighbours, sigma),
                _knn_edge_cfg("truncation", data_node_name, num_nearest_neighbours, sigma),
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

    Creates a fresh subgraph that shares data-node coordinates by reference
    (no copy/clone) with *graph_data* and adds KNN self-loop edges.
    Edge name is ``(data_node_name, "to", data_node_name)``.

    Parameters
    ----------
    graph_data:
        Main graph; data-node coordinates are read from here by reference.
    data_node_name:
        Node type in *graph_data* that holds the data-grid coordinates.
    smoother_config:
        Single-smoother mapping with keys: ``num_nearest_neighbours`` and
        ``sigma``. Edge weights use ``gauss_weight`` with ``l1``-normalised
        Gaussian distances.

    Returns
    -------
    HeteroData
        Subgraph with self-loop KNN smoother edges over *data_node_name*.
    """
    from anemoi.graphs.create import GraphCreator

    if OmegaConf.is_config(smoother_config):
        smoother_config = OmegaConf.to_container(smoother_config, resolve=True)

    num_nearest_neighbours = smoother_config["num_nearest_neighbours"]
    sigma = smoother_config["sigma"]

    subgraph = HeteroData()
    # Share tensors by reference — no copy of the underlying coordinate data.
    subgraph[data_node_name].x = graph_data[data_node_name].x
    subgraph[data_node_name].num_nodes = graph_data[data_node_name].num_nodes

    config = OmegaConf.create(
        {
            "nodes": {},
            "edges": [
                _knn_edge_cfg(data_node_name, data_node_name, num_nearest_neighbours, sigma),
            ],
        }
    )

    return GraphCreator(config).update_graph(subgraph)
