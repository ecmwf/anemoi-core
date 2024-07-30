import logging
from itertools import chain
from pathlib import Path
from typing import Optional
from typing import Union

import torch
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)


class GraphCreator:
    """Graph creator."""

    def __init__(
        self,
        config: Union[Path, DotDict],
    ):
        self.config = DotDict.from_file(config) if isinstance(config, Path) else config

    def generate_graph(self) -> HeteroData:
        """Generate the graph.

        It instantiates the node builders and edge builders defined in the configuration
        file and applies them to the graph.

        Returns
        -------
            HeteroData: The generated graph.
        """
        graph = HeteroData()

        for nodes_name, nodes_cfg in self.config.nodes.items():
            graph = instantiate(nodes_cfg.node_builder, name=nodes_name).update_graph(
                graph, nodes_cfg.get("attributes", {})
            )

        for edges_cfg in self.config.edges:
            graph = instantiate(edges_cfg.edge_builder, edges_cfg.source_name, edges_cfg.target_name).update_graph(
                graph, edges_cfg.get("attributes", {})
            )

        return graph

    def clean(self, graph: HeteroData) -> HeteroData:
        """Remove private attributes used during creation from the graph.

        Parameters
        ----------
        graph : HeteroData
            generated graph

        Returns
        -------
        HeteroData
            cleaned graph
        """
        for type_name in chain(graph.node_types, graph.edge_types):
            for attr_name in graph[type_name].keys():
                if attr_name.startswith("_"):
                    del graph[type_name][attr_name]

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
            LOGGER.info("Graph already exists. Use overwrite=True to overwrite.")

    def create(self, save_path: Optional[Path] = None, overwrite: bool = False) -> HeteroData:
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

        graph = self.generate_graph()
        graph = self.clean(graph)

        if save_path is None:
            LOGGER.warning("No output path specified. The graph will not be saved.")
        else:
            self.save(graph, save_path, overwrite)

        return graph
