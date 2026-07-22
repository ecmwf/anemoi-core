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
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.graphs.edges.builders.base import BaseEdgeBuilder
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder
from anemoi.graphs.processors.post_process import PostProcessor
from anemoi.utils.parametrisation import Parametrisation

LOGGER = logging.getLogger(__name__)


class GraphBuilder:
    """Create a graph using without a config."""

    def __init__(
        self,
        nodes: list[BaseNodeBuilder] | None = None,
        edges: list[BaseEdgeBuilder] | None = None,
        post_processors: list[PostProcessor] | None = None,
    ):
        self.nodes = nodes or []
        self.edges = edges or []
        self.post_processors = post_processors or []

    def update_graph(self, graph: HeteroData) -> HeteroData:
        """Update the graph.

        It iterates over the node builders and edge builders and applies them to the graph.

        Parameters
        ----------
        graph : HeteroData
            The input graph to be updated.

        Returns
        -------
        HeteroData
            The updated graph with new nodes and edges added.
        """
        for node in self.nodes:
            graph = node.update_graph(graph)

        for edge in self.edges:
            graph = edge.update_graph(graph)

        if graph.num_nodes == 0:
            LOGGER.warning("The graph that was created has no nodes.")

        return graph

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
        """Allow post-processing of the resulting graph.

        This method applies any post-processors to the graph,
        which can modify or enhance the graph structure or attributes.

        Parameters
        ----------
        graph : HeteroData
            The graph to be post-processed.

        Returns
        -------
        HeteroData
            The post-processed graph.
        """
        for processor in self.post_processors:
            graph = processor.update_graph(graph)

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
        """Create the graph and save it to the output path.

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
    """Create a graph from a configuration file."""

    def __init__(self, config: str | Path | dict | DictConfig | Parametrisation):
        # Normalise the recipe into a Parametrisation: it both holds the config (``get``) and
        # builds the node/edge/attribute objects from their ``_target_`` specs
        # (``create_module``). Files and OmegaConf configs are normalised to a plain mapping.
        match config:
            case Parametrisation():
                params = config
            case str() | Path():
                params = Parametrisation.from_dict(OmegaConf.to_container(OmegaConf.load(config), resolve=True))
            case DictConfig():
                params = Parametrisation.from_dict(OmegaConf.to_container(config, resolve=True))
            case _:
                params = Parametrisation.from_dict(dict(config))

        self.params = params

        super().__init__(
            nodes=_parse_nodes(params),
            edges=_parse_edges(params),
            post_processors=_parse_post_processors(params),
        )


def _parse_nodes(params: Parametrisation) -> list[BaseNodeBuilder]:
    _nodes = []
    nodes_cfg = params.get("nodes", None)
    if nodes_cfg:
        for node_name, node_cfg in nodes_cfg.items():
            node_builder_cfg = node_cfg["node_builder"]
            attributes_cfg = node_cfg.get("attributes") or {}

            attributes = [
                params.create_module(attr_cfg, name=attr_name) for attr_name, attr_cfg in attributes_cfg.items()
            ]

            node = params.create_module(node_builder_cfg, name=node_name, attributes=attributes)
            _nodes.append(node)
    return _nodes


def _parse_edges(params: Parametrisation) -> list[BaseEdgeBuilder]:
    _edges = []
    edges_cfg = params.get("edges", None)
    if edges_cfg:
        for edge_cfg in edges_cfg:
            source_name = edge_cfg["source_name"]
            target_name = edge_cfg["target_name"]
            source_mask_attr_name = edge_cfg.get("source_mask_attr_name")
            target_mask_attr_name = edge_cfg.get("target_mask_attr_name")
            attributes_cfg = edge_cfg.get("attributes") or {}

            attributes = [
                params.create_module(attr_cfg, name=attr_name) for attr_name, attr_cfg in attributes_cfg.items()
            ]

            # Each edge can have multiple edge builders
            for builder_cfg in edge_cfg["edge_builders"]:
                edge_builder = params.create_module(
                    builder_cfg,
                    source_name=source_name,
                    target_name=target_name,
                    source_mask_attr_name=source_mask_attr_name,
                    target_mask_attr_name=target_mask_attr_name,
                    attributes=attributes,  # Pass attributes to each builder
                )
                _edges.append(edge_builder)
    return _edges


def _parse_post_processors(params: Parametrisation) -> list[PostProcessor]:
    _post_processors = []
    post_processors_cfg = params.get("post_processors", None)
    if post_processors_cfg:
        for pp_cfg in post_processors_cfg:
            post_processor = params.create_module(pp_cfg)
            _post_processors.append(post_processor)
    return _post_processors


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
