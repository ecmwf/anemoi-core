# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
import time
from abc import ABC
from abc import abstractmethod

import torch
from hydra.utils import instantiate
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.utils import concat_edges
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class BaseEdgeBuilder(ABC):
    """Base class for edge builders."""

    def __init__(
        self,
        source_name: str,
        target_name: str,
        source_mask_attr_name: str | None = None,
        target_mask_attr_name: str | None = None,
    ):
        self.source_name = source_name
        self.target_name = target_name
        self.source_mask_attr_name = source_mask_attr_name
        self.target_mask_attr_name = target_mask_attr_name

    @property
    def name(self) -> tuple[str, str, str]:
        """Name of the edge subgraph."""
        return self.source_name, "to", self.target_name

    @abstractmethod
    def compute_edge_index(self, source_nodes: NodeStorage, target_nodes: NodeStorage) -> torch.Tensor: ...

    def prepare_node_data(self, graph: HeteroData) -> tuple[NodeStorage, NodeStorage]:
        """Prepare node information and get source and target nodes."""
        return graph[self.source_name], graph[self.target_name]

    def get_edge_index(self, graph: HeteroData) -> torch.Tensor:
        """Get the edge index."""
        source_nodes, target_nodes = self.prepare_node_data(graph)
        return self.compute_edge_index(source_nodes, target_nodes)

    def register_edges(self, graph: HeteroData) -> HeteroData:
        """Register edges in the graph.

        Parameters
        ----------
        graph : HeteroData
            The graph to register the edges.

        Returns
        -------
        HeteroData
            The graph with the registered edges.
        """
        edge_index = self.get_edge_index(graph)
        edge_type = type(self).__name__

        if "edge_index" in graph[self.name]:
            # Expand current edge indices
            graph[self.name].edge_index = concat_edges(graph[self.name].edge_index, edge_index)
            if edge_type not in graph[self.name].edge_type:
                graph[self.name].edge_type = graph[self.name].edge_type + "," + edge_type
            return graph

        # Register new edge indices
        graph[self.name].edge_index = edge_index
        graph[self.name].edge_type = edge_type
        return graph

    def register_attributes(self, graph: HeteroData, config: DotDict) -> HeteroData:
        """Register attributes in the edges of the graph specified.

        Parameters
        ----------
        graph : HeteroData
            The graph to register the attributes.
        config : DotDict
            The configuration of the attributes.

        Returns
        -------
        HeteroData
            The graph with the registered attributes.
        """
        for attr_name, attr_config in config.items():
            edge_index = graph[self.name].edge_index
            source_coords = graph[self.name[0]].x
            target_coords = graph[self.name[2]].x
            edge_builder = instantiate(attr_config)
            graph[self.name][attr_name] = edge_builder(x=(source_coords, target_coords), edge_index=edge_index)
        return graph

    def update_graph(self, graph: HeteroData, attrs_config: DotDict | None = None) -> HeteroData:
        """Update the graph with the edges.

        Parameters
        ----------
        graph : HeteroData
            The graph.
        attrs_config : DotDict
            The configuration of the edge attributes.

        Returns
        -------
        HeteroData
            The graph with the edges.
        """
        if torch.cuda.is_available():
            graph = graph.to("cuda")

        t0 = time.time()
        graph = self.register_edges(graph)
        t1 = time.time()
        LOGGER.debug("Time to register edge indices (%s): %.2f s", self.__class__.__name__, t1 - t0)

        if attrs_config is not None:
            t0 = time.time()
            graph = self.register_attributes(graph, attrs_config)
            t1 = time.time()
            LOGGER.debug("Time to register edge attribute (%s): %.2f s", self.__class__.__name__, t1 - t0)

        return graph
