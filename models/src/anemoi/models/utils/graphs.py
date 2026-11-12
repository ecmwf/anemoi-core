# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import Mapping
from pathlib import Path

from omegaconf import ListConfig
from torch_geometric.data import HeteroData

from anemoi.graphs.create import GraphCreator
from anemoi.graphs.utils import load_graph_from_file
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


def hidden_node_names(hidden_nodes_name: str | list[str] | Mapping[str, str]) -> list[str]:
    """Names of the hidden meshes, in order and without repeats.

    Parameters
    ----------
    hidden_nodes_name : str, list of str or mapping of str to str
        One hidden mesh, the levels of a hierarchical model, or a mapping from dataset
        name to the hidden mesh it is encoded into.

    Returns
    -------
    list[str]
        The hidden mesh names.
    """
    if isinstance(hidden_nodes_name, str):
        return [hidden_nodes_name]
    if isinstance(hidden_nodes_name, Mapping):
        return list(dict.fromkeys(hidden_nodes_name.values()))
    if isinstance(hidden_nodes_name, (list, tuple, ListConfig)):
        return list(dict.fromkeys(hidden_nodes_name))
    msg = f"Hidden nodes name must be a string, a list of strings, or a dataset mapping, got {type(hidden_nodes_name)}"
    raise TypeError(msg)


def split_graph_config(
    graph_config: DotDict,
    is_dataset_static: dict[str, bool],
    hidden_nodes_name: str | list[str] | Mapping[str, str],
) -> tuple[DotDict, DotDict]:
    """This function creates the static graph structure and returns the dictionary for the dynamic graph configuration.

    Parameters
    ----------
    graph_config : DotDict
        Graph configuration.
    is_dataset_static : dict[str, bool]
        Dictionary indicating whether each dataset is static (e.g., static grid) or not.
    hidden_nodes_name : str, list of str or mapping of str to str
        Name(s) of the hidden nodes in the graph, or a mapping from dataset name to hidden
        nodes name. The hidden nodes are considered to be static.

    Returns
    -------
    tuple[DotDict, DotDict]
        Static graph configuration and dynamic graph configuration.
    """
    for hidden_name in hidden_node_names(hidden_nodes_name):
        is_dataset_static[hidden_name] = True

    static_graph_config = {"nodes": {}, "edges": [], "post_processors": graph_config.get("post_processors") or []}
    dynamic_graph_config = {"nodes": {}, "edges": {}}
    for nodes_name, nodes_config in (graph_config.get("nodes") or {}).items():
        if is_dataset_static.get(nodes_name, True):
            static_graph_config["nodes"][nodes_name] = nodes_config
        else:
            dynamic_graph_config["nodes"][nodes_name] = nodes_config

    for edge_config in graph_config.get("edges") or []:
        source_name = edge_config.source_name
        target_name = edge_config.target_name
        if is_dataset_static.get(source_name, True) and is_dataset_static.get(target_name, True):
            static_graph_config["edges"].append(edge_config)
            dynamic_graph_config["edges"][(source_name, "to", target_name)] = {}
        else:
            dynamic_graph_config["edges"][(source_name, "to", target_name)] = {
                "edge_builders": edge_config.edge_builders,
                "attributes": edge_config.attributes,
            }

    return DotDict(static_graph_config), DotDict(dynamic_graph_config)


def get_missing_static_config(graph: HeteroData, static_graph_config: DotDict) -> DotDict:
    """Return the static nodes and edges from the config that are missing in the graph."""
    return DotDict(
        {
            "nodes": {name: cfg for name, cfg in static_graph_config.nodes.items() if name not in graph.node_types},
            "edges": [
                cfg
                for cfg in static_graph_config.edges
                if (cfg.source_name, "to", cfg.target_name) not in graph.edge_types
            ],
        }
    )


def build_graph(
    graph_config: DotDict,
    is_dataset_static: dict[str, bool],
    hidden_nodes_name: str | list[str] | Mapping[str, str],
    graph_path: str | Path | None = None,
) -> tuple[HeteroData, DotDict]:
    """Build the static graph and the dynamic graph configuration.

    1. Load the graph stored at ``graph_path``, if it exists and ``overwrite`` is not set.
    2. Split the graph config into its static and dynamic parts.
    3. Build the static nodes and edges missing from the loaded graph. When nothing was
       loaded, the whole static graph is created (post-processors included) and saved to
       ``graph_path``.

    Parameters
    ----------
    graph_config : DotDict
        Graph configuration. It may define no nodes or edges when the graph is loaded from file.
    is_dataset_static : dict[str, bool]
        Whether each dataset has a static grid. The hidden nodes are added as static.
    hidden_nodes_name : str, list of str or mapping of str to str
        Name(s) of the hidden nodes in the graph, or a mapping from dataset name to hidden
        nodes name.
    graph_path : str | Path | None, optional
        Where the static graph is loaded from or saved to.

    Returns
    -------
    tuple[HeteroData, DotDict]
        The static graph and the dynamic graph configuration.
    """
    is_dataset_static = dict(is_dataset_static)
    graph_path: Path | None = Path(graph_path) if graph_path is not None else None

    # 1. Load the stored graph.
    graph = None
    overwrite = graph_config.get("overwrite", False)
    if graph_path is not None and graph_path.exists() and not overwrite:
        graph = load_graph_from_file(graph_path)

    # 2. Split the graph config into its static and dynamic parts.
    static_graph_config, dynamic_graph_config = split_graph_config(graph_config, is_dataset_static, hidden_nodes_name)

    # 3. Build the static parts that were not loaded.
    if graph is None:
        if not static_graph_config.nodes:
            msg = (
                "The graph config defines no static nodes and no pre-built graph was found at "
                f"`system.input.graph` ({graph_path})."
            )
            raise ValueError(msg)
        graph = HeteroData()
        graph_config = static_graph_config
    else:
        graph_config = get_missing_static_config(graph, static_graph_config)

    if not graph_config.nodes and not graph_config.edges:
        LOGGER.info("No missing static nodes or edges to build.")
        return graph, dynamic_graph_config

    LOGGER.info(
        "Building the static graph parts missing from the loaded graph: nodes %s, edges %s.",
        list(graph_config.nodes),
        [(cfg.source_name, "to", cfg.target_name) for cfg in graph_config.edges],
    )
    graph_creator = GraphCreator(graph_config)
    graph = graph_creator.update_graph(graph)
    graph = graph_creator.clean(graph)
    graph = graph_creator.post_process(graph)
    if graph_path is not None:
        graph_creator.save(graph, save_path=graph_path, overwrite=overwrite)

    return graph, dynamic_graph_config
