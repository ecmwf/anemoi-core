import logging
import os

import torch
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)


class GraphCreator:
    """Graph creator."""

    def __init__(
        self,
        path,
        config=None,
        cache=None,
        print=print,
        overwrite=False,
        **kwargs,
    ):
        if isinstance(config, str) or isinstance(config, os.PathLike):
            self.config = DotDict.from_file(config)
        else:
            self.config = config

        self.path = path  # Output path
        self.cache = cache
        self.print = print
        self.overwrite = overwrite

    def init(self):
        if self._path_readable() and not self.overwrite:
            raise Exception(f"{self.path} already exists. Use overwrite=True to overwrite.")

    def generate_graph(self) -> HeteroData:
        """Generate the graph.

        It instantiates the node builders and edge builders defined in the configuration
        file and applies them to the graph.

        Returns
        -------
            HeteroData: The generated graph.
        """
        graph = HeteroData()

        for nodes_cfg in self.config.nodes:
            graph = instantiate(nodes_cfg.node_builder, name=nodes_cfg.name).update_graph(
                graph, nodes_cfg.get("attributes", {})
            )

        for edges_cfg in self.config.edges:
            graph = instantiate(edges_cfg.edge_builder, edges_cfg.source_name, edges_cfg.target_name).update_graph(
                graph, edges_cfg.get("attributes", {})
            )

        return graph

    def save(self, graph: HeteroData) -> None:
        """Save the graph to the output path."""
        if not os.path.exists(self.path) or self.overwrite:
            torch.save(graph, self.path)
            self.print(f"Graph saved at {self.path}.")

    def create(self) -> HeteroData:
        """Create the graph and save it to the output path."""
        self.init()
        graph = self.generate_graph()
        self.save(graph)
        return graph

    def _path_readable(self) -> bool:
        """Check if the output path is readable."""
        import torch

        try:
            torch.load(self.path)
            return True
        except FileNotFoundError:
            return False
