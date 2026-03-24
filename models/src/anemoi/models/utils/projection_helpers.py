# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
"""Helpers for naming projection nodes and edges and looking them up in a graph."""

from collections.abc import Mapping

from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

DEFAULT_DATASET_NAME = "data"
DEFAULT_EDGE_RELATION_NAME = "to"
DEFAULT_EDGE_WEIGHT_ATTRIBUTE = "gauss_weight"


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
    """Return whether the graph already has one node group per dataset.

    In this form, each dataset name is itself a node group in the graph,
    rather than reusing a single generic ``data`` node group.
    """
    if not dataset_names:
        return False
    node_names = get_graph_node_names(graph_or_config)
    if not set(dataset_names).issubset(node_names):
        return False

    return dataset_names != [DEFAULT_DATASET_NAME] or DEFAULT_DATASET_NAME not in node_names


def dataset_projection_node_name(dataset_name: str, base_node_name: str) -> str:
    """Turn a local projection node name into the dataset-specific node name."""
    return dataset_name if base_node_name == DEFAULT_DATASET_NAME else f"{dataset_name}_{base_node_name}"


def projection_node_name(
    base_node_name: str,
    *,
    dataset_name: str,
    graph_or_config: HeteroData | DictConfig | Mapping | None = None,
    dataset_names: list[str] | None = None,
    fused_dataset_graph: bool | None = None,
    prefer_existing: bool = False,
) -> str:
    """Resolve a local projection node name to the actual node name in the graph.

    Projection configs are authored using local names such as ``data``,
    ``truncation``, or ``smooth_4x``. This helper turns those local names into
    the concrete node group present in the target graph.
    """
    if fused_dataset_graph is None:
        assert graph_or_config is not None and dataset_names is not None
        fused_dataset_graph = uses_fused_dataset_graph(graph_or_config, dataset_names)

    candidate = dataset_projection_node_name(dataset_name, base_node_name) if fused_dataset_graph else base_node_name
    if not prefer_existing:
        return candidate

    assert graph_or_config is not None
    node_names = get_graph_node_names(graph_or_config)
    if base_node_name in node_names:
        return base_node_name
    if candidate in node_names:
        return candidate
    return base_node_name


def projection_edge_name(
    source_name: str,
    target_name: str,
    *,
    dataset_name: str,
    graph_or_config: HeteroData | DictConfig | Mapping | None = None,
    dataset_names: list[str] | None = None,
    fused_dataset_graph: bool | None = None,
    relation_name: str = DEFAULT_EDGE_RELATION_NAME,
    prefer_existing: bool = False,
) -> tuple[str, str, str]:
    """Resolve a local projection edge tuple to the actual edge tuple in the graph."""
    return (
        projection_node_name(
            source_name,
            dataset_name=dataset_name,
            graph_or_config=graph_or_config,
            dataset_names=dataset_names,
            fused_dataset_graph=fused_dataset_graph,
            prefer_existing=prefer_existing,
        ),
        relation_name,
        projection_node_name(
            target_name,
            dataset_name=dataset_name,
            graph_or_config=graph_or_config,
            dataset_names=dataset_names,
            fused_dataset_graph=fused_dataset_graph,
            prefer_existing=prefer_existing,
        ),
    )


def residual_projection_truncation_node_name(
    projection_config: Mapping | DictConfig | None = None,
) -> str:
    """Return the configured base node name used for truncation projection nodes.

    This keeps compatibility with older config spellings while defaulting to
    the default local projection name ``truncation``.
    """
    if projection_config is None:
        return "truncation"

    truncation = projection_config.get("truncation")
    if isinstance(truncation, Mapping) or OmegaConf.is_config(truncation):
        truncation = OmegaConf.to_container(truncation, resolve=True) if OmegaConf.is_config(truncation) else truncation
        node_name = truncation.get("node_name")
        if isinstance(node_name, str):
            return node_name
        legacy_name = truncation.get("truncation_nodes")
        if isinstance(legacy_name, str):
            return legacy_name

    node_name = projection_config.get("truncation_node_name")
    if isinstance(node_name, str):
        return node_name

    legacy_name = projection_config.get("truncation_nodes")
    if isinstance(legacy_name, str):
        return legacy_name

    return "truncation"


def residual_projection_edge_names(
    *,
    dataset_name: str,
    graph_or_config: HeteroData | DictConfig | Mapping,
    dataset_names: list[str],
    truncation_projection_config: Mapping | DictConfig | None = None,
) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    """Return the down/up edge tuples used by ``TruncatedConnection``.

    The returned names already follow the multi-dataset graph naming when the
    graph contains one node group per dataset.
    """
    truncation_node_name = residual_projection_truncation_node_name(truncation_projection_config)
    relation_name = (
        truncation_projection_config.get("relation_name", DEFAULT_EDGE_RELATION_NAME)
        if truncation_projection_config is not None
        else DEFAULT_EDGE_RELATION_NAME
    )
    down_edges = projection_edge_name(
        DEFAULT_DATASET_NAME,
        truncation_node_name,
        dataset_name=dataset_name,
        graph_or_config=graph_or_config,
        dataset_names=dataset_names,
        relation_name=relation_name,
    )
    up_edges = projection_edge_name(
        truncation_node_name,
        DEFAULT_DATASET_NAME,
        dataset_name=dataset_name,
        graph_or_config=graph_or_config,
        dataset_names=dataset_names,
        relation_name=relation_name,
    )
    return down_edges, up_edges


def multiscale_loss_matrices_graph(
    projection_config: Mapping | DictConfig | None,
    *,
    dataset_name: str,
    graph_or_config: HeteroData | DictConfig | Mapping,
    dataset_names: list[str],
) -> list[dict[str, object] | None] | None:
    """Build ``loss_matrices_graph`` entries from the multiscale smoother config.

    Returns the list used by ``MultiscaleLossWrapper``: one graph edge
    reference per smoother, followed by ``None`` for the full-resolution scale.
    """
    if projection_config is None:
        return None

    smoothers = projection_config.get("smoothers")
    if not smoothers:
        return None

    if OmegaConf.is_config(smoothers):
        smoothers = OmegaConf.to_container(smoothers, resolve=True)

    matrices: list[dict[str, object] | None] = []
    for smoother_name, smoother_cfg in reversed(list(smoothers.items())):
        smoother_cfg = (
            OmegaConf.to_container(smoother_cfg, resolve=True)
            if OmegaConf.is_config(smoother_cfg)
            else dict(smoother_cfg)
        )
        node_name = smoother_cfg.get("node_name", smoother_name)
        edge_weight_attribute = smoother_cfg.get("edge_weight_attribute", DEFAULT_EDGE_WEIGHT_ATTRIBUTE)
        row_normalize = bool(smoother_cfg.get("row_normalize", False))
        src_node_weight_attribute = smoother_cfg.get("src_node_weight_attribute")

        edges_name = projection_edge_name(
            node_name,
            node_name,
            dataset_name=dataset_name,
            graph_or_config=graph_or_config,
            dataset_names=dataset_names,
            relation_name=smoother_cfg.get("relation_name", DEFAULT_EDGE_RELATION_NAME),
        )

        entry: dict[str, object] = {
            "edges_name": list(edges_name),
            "edge_weight_attribute": edge_weight_attribute,
        }
        if src_node_weight_attribute is not None:
            entry["src_node_weight_attribute"] = src_node_weight_attribute
        if row_normalize:
            entry["row_normalize"] = row_normalize

        matrices.append(entry)

    matrices.append(None)
    return matrices
