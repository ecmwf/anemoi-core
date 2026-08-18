# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from itertools import chain
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from anemoi.utils.builder import as_dict
from anemoi.utils.builder import build
from anemoi.utils.builder import build_all
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class GraphBuilder:
    """Build a graph from already-constructed node/edge builders and post-processors.

    This is the object-injection API: rather than reading a configuration, it is handed
    fully-built ``BaseNodeBuilder``/``BaseEdgeBuilder``/``PostProcessor`` objects (each
    carrying its own attributes) and orchestrates graph creation from them. It is the
    shared base for the config-driven :class:`GraphCreator`.

    Parameters
    ----------
    nodes : list, optional
        Built node builders. Each is applied in order via ``update_graph``.
    edges : list, optional
        Built edge builders. Each is applied in order via ``update_graph``.
    post_processors : list, optional
        Built post-processors applied after cleaning.
    """

    def __init__(self, nodes: list | None = None, edges: list | None = None, post_processors: list | None = None):
        self.nodes = list(nodes or [])
        self.edges = list(edges or [])
        self.post_processors = list(post_processors or [])

    def update_graph(self, graph: HeteroData) -> HeteroData:
        """Register the nodes and edges of every builder onto ``graph``."""
        for node_builder in self.nodes:
            graph = node_builder.update_graph(graph)
        for edge_builder in self.edges:
            graph = edge_builder.update_graph(graph)

        if graph.num_nodes == 0:
            LOGGER.warning("The graph that was created has no nodes. Please check your graph configuration.")

        return graph

    def graph_config(self) -> dict:
        """Synthesize the ``graph_config`` mapping post-processors expect.

        Post-processors that recompute edge attributes look them up by source/target
        node names; here they are provided directly as the built attribute objects.
        """
        return {
            "edges": [
                {
                    "source_name": edge_builder.source_name,
                    "target_name": edge_builder.target_name,
                    "attributes": edge_builder.attributes,
                }
                for edge_builder in self.edges
            ]
        }

    def clean(self, graph: HeteroData) -> HeteroData:
        """Remove private attributes used during creation from the graph.

        Parameters
        ----------
        graph : HeteroData
            Generated graph

        Returns
        -------
        HeteroData
            Cleaned graph
        """
        LOGGER.info("Cleaning graph.")
        for type_name in chain(graph.node_types, graph.edge_types):
            attr_names_to_remove = [attr_name for attr_name in graph[type_name] if attr_name.startswith("_")]
            for attr_name in attr_names_to_remove:
                del graph[type_name][attr_name]
                LOGGER.info(f"{attr_name} deleted from graph.")

        return graph

    def post_process(self, graph: HeteroData) -> HeteroData:
        """Apply the post-processors to the graph, in order."""
        graph_config = self.graph_config()
        for processor in self.post_processors:
            graph = processor.update_graph(graph, graph_config=graph_config)

        return graph

    def save(self, graph: HeteroData, save_path: Path, overwrite: bool = False) -> None:
        """Save the generated graph to the output path.

        Parameters
        ----------
        graph : HeteroData
            generated graph
        save_path : Path
            location to save the graph
        overwrite : bool, optional
            whether to overwrite existing graph file, by default False
        """
        save_path = Path(save_path)

        if not save_path.exists() or overwrite:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(graph, save_path)
            LOGGER.info(f"Graph saved at {save_path}.")
        else:
            # The error is only logged for compatibility with multi-gpu training in anemoi-training.
            # Currently, distributed graph creation is not supported so we create the same graph in each gpu.
            LOGGER.error(
                f"Graph not saved because {save_path} already exists. If this occurred during a multi-process or multi-GPU run, another process likely saved it first. If you intended to recreate it, rerun with overwrite=True."
            )

    def create(self, save_path: Path | None = None, overwrite: bool = False) -> HeteroData:
        """Create the graph and optionally save it to the output path.

        Parameters
        ----------
        save_path : Path, optional
            location to save the graph, by default None
        overwrite : bool, optional
            whether to overwrite existing graph file, by default False

        Returns
        -------
        HeteroData
            created graph object
        """
        graph = HeteroData()
        graph = self.update_graph(graph)
        graph = self.clean(graph)
        graph = self.post_process(graph)

        if save_path is None:
            LOGGER.warning("No output path specified. The graph will not be saved.")
        else:
            self.save(graph, save_path, overwrite)

        return graph


class GraphCreator(GraphBuilder):
    """Graph creator that builds the graph from a (Hydra-style) configuration."""

    config: DotDict

    def __init__(
        self,
        config: str | Path | DotDict | DictConfig,
    ):
        super().__init__()
        if isinstance(config, Path) or isinstance(config, str):
            self.config = DotDict.from_file(config)
        elif isinstance(config, DictConfig):
            self.config = DotDict(config)
        else:
            self.config = config

    def update_graph(self, graph: HeteroData) -> HeteroData:
        """Update the graph.

        It builds the node builders and edge builders defined in the configuration
        and applies them to the graph.

        Parameters
        ----------
        graph : HeteroData
            The input graph to be updated.

        Returns
        -------
        HeteroData
            The updated graph with new nodes and edges added based on the configuration.
        """
        config = as_dict(self.config)

        for nodes_name, nodes_cfg in config.get("nodes", {}).items():
            node_builder = build(nodes_cfg["node_builder"], name=nodes_name)
            attributes = build_all(nodes_cfg.get("attributes", {}) or {})
            graph = node_builder.update_graph(graph, attributes)

        for edges_cfg in config.get("edges", []) or []:
            edge_builders = [
                build(edge_builder_cfg, source_name=edges_cfg["source_name"], target_name=edges_cfg["target_name"])
                for edge_builder_cfg in edges_cfg["edge_builders"]
            ]
            for edge_builder in edge_builders:
                graph = edge_builder.register_edges(graph)

            attributes = build_all(edges_cfg.get("attributes", {}) or {})
            graph = edge_builders[-1].register_attributes(graph, attributes)

        if graph.num_nodes == 0:
            LOGGER.warning("The graph that was created has no nodes. Please check your graph configuration file.")

        return graph

    def post_process(self, graph: HeteroData) -> HeteroData:
        """Apply the post-processors defined in the configuration, in order.

        Each post-processor is built from config and receives the (plain-dict) graph
        configuration so that it can, e.g., recompute edge attributes after masking.
        """
        graph_config = as_dict(self.config)
        for processor in graph_config.get("post_processors", []) or []:
            graph = build(processor).update_graph(graph, graph_config=graph_config)

        return graph


def load_graph_from_file(graph_filename: Path) -> HeteroData:
    """Load a serialized graph on the currently active distributed device."""
    try:
        from anemoi.graphs.utils import get_distributed_device

        map_location = get_distributed_device()
    except Exception:
        map_location = "cpu"

    LOGGER.info("Loading graph data from %s", graph_filename)
    return torch.load(graph_filename, map_location=map_location, weights_only=False)


def validate_loaded_graph(graph_data: HeteroData, required_dataset_names: list[str]) -> None:
    """Ensure the loaded graph contains the required dataset node types."""
    missing = [n for n in required_dataset_names if n not in graph_data.node_types]
    if missing:
        msg = (
            "Loaded graph is missing dataset node types required by the dataloader. "
            f"Missing {missing}; available nodes are {graph_data.node_types}."
        )
        raise ValueError(msg)
