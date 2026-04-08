# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Utilities for expanding projection shorthand configs into graph nodes and edges."""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf
from omegaconf import open_dict

from anemoi.graphs.projection_helpers import DEFAULT_DATASET_NAME
from anemoi.graphs.projection_helpers import DEFAULT_EDGE_RELATION_NAME
from anemoi.graphs.projection_helpers import DEFAULT_EDGE_WEIGHT_ATTRIBUTE
from anemoi.graphs.projection_helpers import dataset_projection_node_name
from anemoi.graphs.projection_helpers import expand_geometric_smoothers
from anemoi.graphs.projection_helpers import projection_node_name
from anemoi.graphs.projection_helpers import residual_projection_truncation_node_name
from anemoi.graphs.projection_helpers import uses_fused_dataset_graph


def _to_container(value: Any, *, resolve: bool) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=resolve)
    return value


def _merge_dataset_projection_overrides(projection_cfg: Any, dataset_name: str | None) -> Any:
    if dataset_name is None:
        return projection_cfg
    dataset_overrides = projection_cfg.get("datasets", None)
    if not dataset_overrides or dataset_name not in dataset_overrides:
        return projection_cfg
    return OmegaConf.merge(projection_cfg, dataset_overrides[dataset_name])


def _projection_builds_per_dataset(projection_name: str, projection_cfg: Any) -> bool:
    if projection_cfg.get("build_per_dataset", None) is not None:
        return bool(projection_cfg.build_per_dataset)
    return projection_name in {"multiscale", "truncation"}


def _projection_node_rename_map(projection_cfg: Any, dataset_name: str) -> dict[str, str]:
    rename_map = {DEFAULT_DATASET_NAME: dataset_name}

    data_nodes_name = projection_cfg.get("data_nodes", None)
    if isinstance(data_nodes_name, str):
        rename_map[data_nodes_name] = dataset_name

    projection_nodes = _to_container(projection_cfg.get("nodes", {}), resolve=True)
    for node_name in projection_nodes:
        rename_map[node_name] = dataset_projection_node_name(dataset_name, node_name)

    smoothers = _to_container(projection_cfg.get("smoothers", {}), resolve=True)
    for smoother_name, smoother_cfg in smoothers.items():
        node_name = smoother_cfg.get("node_name", smoother_name)
        if isinstance(node_name, str):
            rename_map[node_name] = dataset_projection_node_name(dataset_name, node_name)

    truncation_node_name = residual_projection_truncation_node_name(projection_cfg)
    if isinstance(truncation_node_name, str):
        rename_map[truncation_node_name] = dataset_projection_node_name(dataset_name, truncation_node_name)

    return rename_map


def _rename_projection_values(value: Any, rename_map: dict[str, str]) -> Any:
    if isinstance(value, str):
        return rename_map.get(value, value)
    if isinstance(value, list):
        return [_rename_projection_values(item, rename_map) for item in value]
    if isinstance(value, tuple):
        return tuple(_rename_projection_values(item, rename_map) for item in value)
    if isinstance(value, dict):
        return {key: _rename_projection_values(item, rename_map) for key, item in value.items()}
    return value


def _expand_projection_for_datasets(projection_cfg: Any, dataset_names: list[str]) -> tuple[dict[str, Any], list[Any]]:
    projection_nodes: dict[str, Any] = {}
    projection_edges: list[Any] = []
    for dataset_name in dataset_names:
        dataset_projection_cfg = _merge_dataset_projection_overrides(projection_cfg, dataset_name)
        rename_map = _projection_node_rename_map(dataset_projection_cfg, dataset_name)
        dataset_nodes = _to_container(dataset_projection_cfg.get("nodes", {}), resolve=True)
        dataset_edges = _to_container(dataset_projection_cfg.get("edges", []), resolve=True)
        projection_nodes.update(
            {
                rename_map.get(node_name, node_name): _rename_projection_values(node_cfg, rename_map)
                for node_name, node_cfg in dataset_nodes.items()
            },
        )
        projection_edges.extend(_rename_projection_values(dataset_edges, rename_map))
    return projection_nodes, projection_edges


def _as_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def _projection_edge_builders(edge_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    builders = _as_list(_to_container(edge_cfg.get("edge_builders"), resolve=True))
    if builders:
        return builders
    builder = edge_cfg.get("edge_builder")
    builders = _as_list(_to_container(builder, resolve=True))
    if builders:
        return builders
    num_nearest_neighbours = edge_cfg.get("num_nearest_neighbours")
    if num_nearest_neighbours is None:
        msg = "Projection edge configuration must define edge_builders, edge_builder, or num_nearest_neighbours."
        raise ValueError(msg)
    return [{"_target_": "anemoi.graphs.edges.KNNEdges", "num_nearest_neighbours": num_nearest_neighbours}]


def _projection_edge_attributes(edge_cfg: dict[str, Any], edge_weight_attribute: str | None) -> dict[str, Any] | None:
    attributes = _to_container(edge_cfg.get("attributes"), resolve=True)
    if attributes is not None:
        return attributes
    attributes = _to_container(edge_cfg.get("edge_attributes"), resolve=True)
    if attributes is not None:
        return attributes
    edge_weights = _to_container(edge_cfg.get("edge_weights"), resolve=True)
    if edge_weights is not None:
        return edge_weights if edge_weight_attribute is None else {edge_weight_attribute: edge_weights}
    edge_weight = _to_container(edge_cfg.get("edge_weight"), resolve=True)
    if edge_weight is not None:
        return edge_weight if edge_weight_attribute is None else {edge_weight_attribute: edge_weight}
    gaussian_norm = edge_cfg.get("gaussian_norm")
    sigma = edge_cfg.get("sigma")
    if gaussian_norm is None or sigma is None:
        return None
    if edge_weight_attribute is None:
        edge_weight_attribute = DEFAULT_EDGE_WEIGHT_ATTRIBUTE
    return {
        edge_weight_attribute: {
            "_target_": "anemoi.graphs.edges.attributes.GaussianDistanceWeights",
            "norm": gaussian_norm,
            "sigma": sigma,
        },
    }


def _build_projection_edge(
    source_name: str,
    target_name: str,
    *,
    dataset_name: str,
    fused_dataset_graph: bool,
    edge_cfg: dict[str, Any],
) -> dict[str, Any]:
    relation_name = edge_cfg.get("relation_name", DEFAULT_EDGE_RELATION_NAME)
    edge_weight_attribute = edge_cfg.get("edge_weight_attribute", DEFAULT_EDGE_WEIGHT_ATTRIBUTE)
    attributes = _projection_edge_attributes(edge_cfg, edge_weight_attribute)
    edge = {
        "source_name": projection_node_name(
            source_name, dataset_name=dataset_name, fused_dataset_graph=fused_dataset_graph
        ),
        "target_name": projection_node_name(
            target_name, dataset_name=dataset_name, fused_dataset_graph=fused_dataset_graph
        ),
        "edge_builders": _projection_edge_builders(edge_cfg),
    }
    if relation_name != DEFAULT_EDGE_RELATION_NAME:
        edge["name"] = relation_name
    if attributes is not None:
        edge["attributes"] = attributes
    return edge


def _projection_node_entry(
    node_name: str,
    *,
    projection_cfg: Any,
    dataset_name: str,
    fused_dataset_graph: bool,
    node_builder_cfg: Any,
    default_node_builder: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    graph_node_name = projection_node_name(
        node_name, dataset_name=dataset_name, fused_dataset_graph=fused_dataset_graph
    )
    node_builder = _to_container(node_builder_cfg, resolve=True)
    if node_builder is None:
        node_builder = default_node_builder
    else:
        node_builder = _rename_projection_values(
            node_builder, _projection_node_rename_map(projection_cfg, dataset_name)
        )
    return graph_node_name, {"node_builder": node_builder}


def _derived_residual_projection(
    projection_cfg: Any,
    *,
    dataset_name: str,
    fused_dataset_graph: bool,
) -> tuple[dict[str, Any], list[Any]] | None:
    truncation_cfg = projection_cfg.get("truncation")
    if truncation_cfg is None:
        return None
    truncation_cfg = _to_container(truncation_cfg, resolve=True)
    truncation_node_name = residual_projection_truncation_node_name(projection_cfg)
    grid = truncation_cfg.get("grid", truncation_cfg.get("truncation_grid"))
    if truncation_cfg.get("node_builder") is None and grid is None:
        msg = "Residual truncation projection requires either truncation.node_builder or truncation.grid."
        raise ValueError(msg)
    graph_node_name, node_entry = _projection_node_entry(
        truncation_node_name,
        projection_cfg=projection_cfg,
        dataset_name=dataset_name,
        fused_dataset_graph=fused_dataset_graph,
        node_builder_cfg=truncation_cfg.get("node_builder"),
        default_node_builder={"_target_": "anemoi.graphs.nodes.ReducedGaussianGridNodes", "grid": grid},
    )
    nodes = {graph_node_name: node_entry}
    edges = [
        _build_projection_edge(
            "data",
            truncation_node_name,
            dataset_name=dataset_name,
            fused_dataset_graph=fused_dataset_graph,
            edge_cfg=truncation_cfg,
        ),
        _build_projection_edge(
            truncation_node_name,
            "data",
            dataset_name=dataset_name,
            fused_dataset_graph=fused_dataset_graph,
            edge_cfg=truncation_cfg,
        ),
    ]
    return nodes, edges


def _derived_multiscale_projection(
    projection_cfg: Any,
    *,
    dataset_name: str,
    fused_dataset_graph: bool,
) -> tuple[dict[str, Any], list[Any]] | None:
    smoothers = projection_cfg.get("smoothers") or expand_geometric_smoothers(
        _to_container(projection_cfg, resolve=True),
    )
    if not smoothers:
        return None
    smoothers = _to_container(smoothers, resolve=True)
    projection_nodes: dict[str, Any] = {}
    projection_edges: list[Any] = []
    for smoother_name, smoother_cfg in smoothers.items():
        smoother_cfg = dict(smoother_cfg)
        node_name = smoother_cfg.get("node_name", smoother_name)
        graph_node_name, node_entry = _projection_node_entry(
            node_name,
            projection_cfg=projection_cfg,
            dataset_name=dataset_name,
            fused_dataset_graph=fused_dataset_graph,
            node_builder_cfg=smoother_cfg.get("node_builder"),
            default_node_builder={
                "_target_": "anemoi.graphs.nodes.ReferenceNodes",
                "reference_node_name": projection_node_name(
                    "data", dataset_name=dataset_name, fused_dataset_graph=fused_dataset_graph
                ),
            },
        )
        projection_nodes[graph_node_name] = node_entry
        projection_edges.append(
            _build_projection_edge(
                node_name,
                node_name,
                dataset_name=dataset_name,
                fused_dataset_graph=fused_dataset_graph,
                edge_cfg=smoother_cfg,
            ),
        )
    return projection_nodes, projection_edges


def _derived_projection_fragments(
    projection_name: str,
    projection_cfg: Any,
    *,
    dataset_name: str,
    fused_dataset_graph: bool,
) -> tuple[dict[str, Any], list[Any]] | None:
    if projection_cfg.get("nodes") is not None or projection_cfg.get("edges") is not None:
        return None
    if projection_name == "truncation":
        return _derived_residual_projection(
            projection_cfg, dataset_name=dataset_name, fused_dataset_graph=fused_dataset_graph
        )
    if projection_name == "multiscale":
        return _derived_multiscale_projection(
            projection_cfg, dataset_name=dataset_name, fused_dataset_graph=fused_dataset_graph
        )
    return None


def _projection_fragments_to_merge(
    projection_name: str,
    projection_cfg: Any,
    *,
    dataset_names: list[str],
    fused_dataset_graph: bool,
) -> tuple[dict[str, Any] | None, list[Any] | None]:
    if fused_dataset_graph and _projection_builds_per_dataset(projection_name, projection_cfg):
        projection_nodes = {}
        projection_edges = []
        for dataset_name in dataset_names:
            dataset_projection_cfg = _merge_dataset_projection_overrides(projection_cfg, dataset_name)
            derived_fragments = _derived_projection_fragments(
                projection_name, dataset_projection_cfg, dataset_name=dataset_name, fused_dataset_graph=True
            )
            if derived_fragments is not None:
                dataset_nodes, dataset_edges = derived_fragments
            else:
                dataset_nodes, dataset_edges = _expand_projection_for_datasets(dataset_projection_cfg, [dataset_name])
            projection_nodes.update(dataset_nodes)
            projection_edges.extend(dataset_edges)
        return projection_nodes, projection_edges

    effective_projection_cfg = _merge_dataset_projection_overrides(
        projection_cfg,
        dataset_names[0] if len(dataset_names) == 1 else None,
    )
    derived_fragments = _derived_projection_fragments(
        projection_name,
        effective_projection_cfg,
        dataset_name=dataset_names[0] if dataset_names else "data",
        fused_dataset_graph=False,
    )
    if derived_fragments is not None:
        return derived_fragments
    return effective_projection_cfg.get("nodes", None), effective_projection_cfg.get("edges", None)


def expand_projections_into_graph_config(graph_config: Any, dataset_names: list[str] | None = None) -> None:
    """Expand the ``projections`` shorthand config into explicit nodes and edges in place.

    Shorthand ``truncation`` and ``multiscale`` configs are expanded first.
    In fused graphs, expansion happens separately for each dataset.
    """
    projections = getattr(graph_config, "projections", None) or {}
    if not projections:
        return

    dataset_names = dataset_names or []
    fused_dataset_graph = uses_fused_dataset_graph(graph_config, dataset_names)

    with open_dict(graph_config):
        if graph_config.get("nodes") is None:
            graph_config.nodes = {}
        if graph_config.get("edges") is None:
            graph_config.edges = []

        for projection_name, projection_cfg in projections.items():
            if projection_cfg is None:
                continue
            projection_nodes, projection_edges = _projection_fragments_to_merge(
                projection_name,
                projection_cfg,
                dataset_names=dataset_names,
                fused_dataset_graph=fused_dataset_graph,
            )
            if projection_nodes:
                graph_config.nodes = {**graph_config.nodes, **projection_nodes}
            if projection_edges:
                graph_config.edges = list(graph_config.edges) + list(projection_edges)
