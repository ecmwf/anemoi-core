# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections.abc import Mapping
from pathlib import Path

from omegaconf import ListConfig
from torch_geometric.data import HeteroData

from anemoi.graphs.utils import load_graph_from_file
from anemoi.graphs.utils import validate_loaded_graph
from anemoi.utils.config import DotDict


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
    msg = (
        "Hidden nodes name must be a string, a list of strings, or a dataset mapping, " f"got {type(hidden_nodes_name)}"
    )
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

    static_graph_config, dynamic_graph_config = {"nodes": {}, "edges": []}, {"nodes": {}, "edges": {}}
    for nodes_name, nodes_config in graph_config.nodes.items():
        if is_dataset_static[nodes_name]:
            static_graph_config["nodes"][nodes_name] = nodes_config
        else:
            dynamic_graph_config["nodes"][nodes_name] = nodes_config

    for edge_config in graph_config.edges:
        source_name = edge_config.source_name
        target_name = edge_config.target_name
        if is_dataset_static[source_name] and is_dataset_static[target_name]:
            static_graph_config["edges"].append(edge_config)
            dynamic_graph_config["edges"][(source_name, "to", target_name)] = {}
        else:
            dynamic_graph_config["edges"][(source_name, "to", target_name)] = {
                "edge_builders": edge_config.edge_builders,
                "attributes": edge_config.attributes,
            }

    return DotDict(static_graph_config), DotDict(dynamic_graph_config)


def load_existing_graph(
    graph_path: str | Path | None,
    is_dataset_static: dict[str, bool],
    hidden_nodes_name: str | list[str] | Mapping[str, str],
) -> tuple[HeteroData, DotDict]:
    """Load a pre-built static graph and return it with an empty dynamic graph configuration.

    This serves the existing-graph mode, where the graph config defines
    no nodes or edges and the graph is read from ``system.input.graph``.

    Parameters
    ----------
    graph_path : str | Path | None
        Path to the serialised graph.
    is_dataset_static : dict[str, bool]
        Dictionary indicating whether each dataset is static. The hidden nodes are added as static.
    hidden_nodes_name : str, list of str or mapping of str to str
        Name(s) of the hidden nodes in the graph, or a mapping from dataset name to hidden
        nodes name.

    Returns
    -------
    tuple[HeteroData, DotDict]
        The loaded graph and a dynamic graph configuration with no builders for any of its edges.
    """
    if graph_path is None:
        msg = "The graph config defines no nodes, so a pre-built graph must be given in `system.input.graph`."
        raise ValueError(msg)
    dynamic_datasets = [name for name, is_static in is_dataset_static.items() if not is_static]
    if dynamic_datasets:
        msg = (
            f"A pre-built graph only works with static datasets, but {dynamic_datasets} require a dynamic graph. "
            "Define their nodes and edges in the graph config instead."
        )
        raise NotImplementedError(msg)

    for hidden_name in hidden_node_names(hidden_nodes_name):
        is_dataset_static[hidden_name] = True

    graph = load_graph_from_file(Path(graph_path))
    validate_loaded_graph(graph, list(is_dataset_static))

    dynamic_graph_config = {"nodes": {}, "edges": {edge_type: {} for edge_type in graph.edge_types}}
    return graph, DotDict(dynamic_graph_config)
