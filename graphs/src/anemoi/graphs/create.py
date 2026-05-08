# (C) Copyright 2024 Anemoi contributors.
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
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class GraphCreator:
    """Graph creator."""

    config: DotDict

    def __init__(
        self,
        config: str | Path | DotDict | DictConfig,
    ):
        if isinstance(config, Path) or isinstance(config, str):
            self.config = DotDict.from_file(config)
        elif isinstance(config, DictConfig):
            self.config = DotDict(config)
        else:
            self.config = config

    def update_graph(self, graph: HeteroData) -> HeteroData:
        """Update the graph.

        It instantiates the node builders and edge builders defined in the configuration
        file and applies them to the graph.

        Parameters
        ----------
        graph : HeteroData
            The input graph to be updated.

        Returns
        -------
        HeteroData
            The updated graph with new nodes and edges added based on the configuration.
        """
        for nodes_name, nodes_cfg in self.config.get("nodes", {}).items():
            graph = instantiate(nodes_cfg.node_builder, name=nodes_name).update_graph(
                graph, attrs_config=nodes_cfg.get("attributes", {})
            )

        for edges_cfg in self.config.get("edges", {}):
            for edge_builder_cfg in edges_cfg.edge_builders:
                edge_builder = instantiate(
                    edge_builder_cfg,
                    source_name=edges_cfg.source_name,
                    target_name=edges_cfg.target_name,
                )
                graph = edge_builder.update_graph(graph, attrs_config=None)

            graph = edge_builder.register_attributes(graph, edges_cfg.get("attributes", {}))

        if graph.num_nodes == 0:
            LOGGER.warning("The graph that was created has no nodes. Please check your graph configuration file.")

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

        This method applies any configured post-processors to the graph,
        which can modify or enhance the graph structure or attributes.

        Parameters
        ----------
        graph : HeteroData
            The graph to be post-processed.

        Returns
        -------
        HeteroData
            The post-processed graph.

        Notes
        -----
        Post-processors are applied in the order they are specified in the configuration.
        Each post-processor should implement an `update_graph` method that takes and returns a HeteroData object.
        """
        for processor in self.config.get("post_processors", []):
            graph = instantiate(processor).update_graph(graph, graph_config=self.config)

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
