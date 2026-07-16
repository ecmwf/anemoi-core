# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)


def _as_dict(value: str | dict | DictConfig) -> str | dict:
    """Convert DictConfig payloads to plain dicts."""
    return dict(value) if isinstance(value, DictConfig) else value


def _normalize_dataset_config(dataset_config: str | dict | DictConfig) -> dict:
    """Normalize dataset payload to the open_dataset dictionary contract."""
    if isinstance(dataset_config, str):
        return {"dataset": dataset_config}

    dataset_config = _as_dict(dataset_config)
    if "dataset" not in dataset_config:
        msg = "dataset_config must contain the 'dataset' key."
        raise ValueError(msg)

    if dataset_config["dataset"] is None:
        msg = "dataset_config.dataset cannot be None."
        raise ValueError(msg)

    invalid_inner_keys = {"start", "end"} & set(dataset_config)
    if invalid_inner_keys:
        invalid = ", ".join(sorted(invalid_inner_keys))
        msg = f"dataset_config cannot contain [{invalid}]. Use outer keys 'start' and 'end' instead."
        raise ValueError(msg)

    # Keep only explicitly set options to avoid passing None-valued kwargs
    # (e.g. select=None), which can trigger downstream subset selection issues.
    return {key: value for key, value in dataset_config.items() if value is not None}


def _normalize_reader_config(dataset_config: dict | DictConfig) -> dict:
    """Validate and normalize reader configuration."""
    normalized = dict(dataset_config)

    if "dataset" in normalized:
        msg = (
            "Invalid dataloader dataset schema: use 'dataset_config' (outer key) "
            "and 'dataset' inside it. The legacy outer 'dataset' key is no longer supported."
        )
        raise ValueError(msg)

    base_dataset_config = normalized.pop("dataset_config", None)
    if base_dataset_config is None:
        msg = "Missing required 'dataset_config' in dataset reader configuration."
        raise ValueError(msg)

    allowed_keys = {"start", "end", "trajectory"}
    unknown_keys = set(normalized) - allowed_keys
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        allowed = ", ".join(sorted({"dataset_config", *allowed_keys}))
        msg = f"Unknown dataset reader option(s) [{unknown}]. Allowed top-level keys: {allowed}."
        raise ValueError(msg)

    normalized["dataset_config"] = base_dataset_config
    return normalized


def get_missing_graph_config(graph_config, current_nodes: list[str], current_edges: list[tuple[str, str, str]]) -> dict:
    missing_graph_config = {"nodes": {}, "edges": []}
    for node in graph_config.nodes:
        if node not in current_nodes:
            # Add missing node to the graph config
            LOGGER.info("Node '%s' not found in the loaded graph.", node)
            missing_graph_config["nodes"][node] = graph_config.nodes[node]

    for edges in graph_config.edges:
        source_nodes = edges.source_nodes
        target_nodes = edges.target_nodes
        if (source_nodes, "to", target_nodes) not in current_edges:
            LOGGER.info("Edge from '%s' to '%s' not found in the loaded graph.", source_nodes, target_nodes)
            # Add missing edge to the graph config
            missing_graph_config["edges"].append(edges)

    return missing_graph_config


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
