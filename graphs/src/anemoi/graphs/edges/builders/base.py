# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import time
from abc import ABC
from abc import abstractmethod

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.edges.builders.masking import NodeMaskingMixin
from anemoi.graphs.utils import PYG_AVAILABLE
from anemoi.graphs.utils import PYG_INSTRUCTIONS
from anemoi.graphs.utils import concat_edges
from anemoi.graphs.utils import get_distributed_device
from anemoi.utils.config import DotDict
from anemoi.utils.parametrisation import DictParametrisation
from anemoi.utils.parametrisation import Parametrisation

LOGGER = logging.getLogger(__name__)

# Default (stateless) parametrisation used to build attribute objects when the caller does
# not supply one -- e.g. builders exercised directly in tests.
_DEFAULT_PARAMETRISATION = DictParametrisation()


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
        self.device = get_distributed_device()

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
        edge_index = self.compute_edge_index(source_nodes, target_nodes)
        return edge_index.to(dtype=torch.int32, device=self.device)

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

    def register_attributes(
        self, graph: HeteroData, config: DotDict, parametrisation: Parametrisation | None = None
    ) -> HeteroData:
        """Register attributes in the edges of the graph specified.

        Parameters
        ----------
        graph : HeteroData
            The graph to register the attributes.
        config : DotDict
            The configuration of the attributes.
        parametrisation : Parametrisation, optional
            Parametrisation used to build each attribute object. Defaults to a stateless
            :class:`~anemoi.utils.parametrisation.DictParametrisation`.

        Returns
        -------
        HeteroData
            The graph with the registered attributes.
        """
        parametrisation = parametrisation or _DEFAULT_PARAMETRISATION
        for attr_name, attr_config in config.items():
            edge_index = graph[self.name].edge_index
            edge_attribute_builder = parametrisation.create_module(attr_config)
            graph[self.name][attr_name] = edge_attribute_builder(
                x=(graph[self.name[0]], graph[self.name[2]]), edge_index=edge_index
            )
        return graph

    def update_graph(
        self,
        graph: HeteroData,
        attrs_config: DotDict | None = None,
        parametrisation: Parametrisation | None = None,
    ) -> HeteroData:
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
        t0 = time.time()
        graph = self.register_edges(graph)
        t1 = time.time()
        LOGGER.debug("Time to register edge indices (%s): %.2f s", self.__class__.__name__, t1 - t0)

        if attrs_config is not None:
            t0 = time.time()
            graph = self.register_attributes(graph, attrs_config, parametrisation)
            t1 = time.time()
            LOGGER.debug("Time to register edge attribute (%s): %.2f s", self.__class__.__name__, t1 - t0)

        return graph


class BaseDistanceEdgeBuilders(BaseEdgeBuilder, NodeMaskingMixin, ABC):
    """Base class for edge builders based on distance."""

    @abstractmethod
    def _compute_edge_index_pyg(self, source_coords: NodeStorage, target_coords: NodeStorage) -> np.ndarray: ...

    @abstractmethod
    def _compute_adj_matrix_sklearn(self, source_coords: NodeStorage, target_coords: NodeStorage) -> np.ndarray: ...

    def compute_edge_index(self, source_nodes: NodeStorage, target_nodes: NodeStorage) -> torch.Tensor:
        """Compute the edge indices.

        Parameters
        ----------
        source_nodes : NodeStorage
            The source nodes.
        target_nodes : NodeStorage
            The target nodes.

        Returns
        -------
        torch.Tensor of shape (2, num_edges)
            Indices of source and target nodes connected by an edge.
        """
        source_coords, target_coords = self.get_cartesian_node_coordinates(source_nodes, target_nodes)

        if PYG_AVAILABLE:
            edge_index = self._compute_edge_index_pyg(source_coords, target_coords)
            edge_index = self.undo_masking_edge_index(edge_index, source_nodes, target_nodes)
        else:
            LOGGER.warning(PYG_INSTRUCTIONS)
            adj_matrix = self._compute_adj_matrix_sklearn(source_coords, target_coords)
            adj_matrix = self.undo_masking_adj_matrix(adj_matrix, source_nodes, target_nodes)
            edge_index = torch.from_numpy(np.stack([adj_matrix.col, adj_matrix.row], axis=0))

        return edge_index
