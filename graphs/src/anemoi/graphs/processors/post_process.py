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
from abc import ABC
from abc import abstractmethod

import torch
from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)


class PostProcessor(ABC):

    @abstractmethod
    def update_graph(self, graph: HeteroData) -> HeteroData:
        raise NotImplementedError(f"The {self.__class__.__name__} class does not implement the method update_graph().")


class BaseMaskingProcessor(PostProcessor, ABC):
    """Base class for mask based processor."""

    def __init__(
        self,
        nodes_name: str,
        save_mask_indices_to_attr: str | None = None,
    ) -> None:
        self.nodes_name = nodes_name
        self.save_mask_indices_to_attr = save_mask_indices_to_attr
        self.mask: torch.Tensor = None

    def removing_nodes(self, graph: HeteroData) -> HeteroData:
        """Remove nodes based on the mask passed."""
        for attr_name in graph[self.nodes_name].node_attrs():
            graph[self.nodes_name][attr_name] = graph[self.nodes_name][attr_name][self.mask]

        return graph

    def create_indices_mapper_from_mask(self) -> dict[int, int]:
        return dict(zip(torch.where(self.mask)[0].tolist(), list(range(self.mask.sum()))))

    def update_edge_indices(self, graph: HeteroData) -> HeteroData:
        """Update the edge indices to the new position of the nodes."""
        idx_mapping = self.create_indices_mapper_from_mask()
        for edges_name in graph.edge_types:
            if edges_name[0] == self.nodes_name:
                graph[edges_name].edge_index[0] = graph[edges_name].edge_index[0].apply_(idx_mapping.get)

            if edges_name[2] == self.nodes_name:
                graph[edges_name].edge_index[1] = graph[edges_name].edge_index[1].apply_(idx_mapping.get)

        return graph

    @abstractmethod
    def compute_mask(self, graph: HeteroData) -> torch.Tensor: ...

    def add_attribute(self, graph: HeteroData) -> HeteroData:
        """Add an attribute of the mask indices as node attribute."""
        if self.save_mask_indices_to_attr is not None:
            LOGGER.info(
                f"An attribute {self.save_mask_indices_to_attr} has been added with the indices to mask the nodes from the original graph."
            )
            mask_indices = torch.where(self.mask)[0].reshape((graph[self.nodes_name].num_nodes, -1))
            graph[self.nodes_name][self.save_mask_indices_to_attr] = mask_indices

        return graph

    def update_graph(self, graph: HeteroData) -> HeteroData:
        """Post-process the graph.

        Parameters
        ----------
        graph: HeteroData
            The graph to post-process.

        Returns
        -------
        HeteroData
            The post-processed graph.
        """
        self.mask = self.compute_mask(graph)
        LOGGER.info(f"Removing {(~self.mask).sum()} nodes from {self.nodes_name}.")
        graph = self.removing_nodes(graph)
        graph = self.update_edge_indices(graph)
        graph = self.add_attribute(graph)
        return graph


class RemoveUnconnectedNodes(BaseMaskingProcessor):
    """Remove unconnected nodes in the graph.

    Attributes
    ----------
    nodes_name: str
        Name of the unconnected nodes to remove.
    ignore: str, optional
        Name of an attribute to ignore when removing nodes. Nodes with
        this attribute set to True will not be removed.
    save_mask_indices_to_attr: str, optional
        Name of the attribute to save the mask indices. If provided,
        the indices of the kept nodes will be saved in this attribute.

    Methods
    -------
    compute_mask(graph)
        Compute the mask of the connected nodes.
    prune_graph(graph, mask)
        Prune the nodes with the specified mask.
    add_attribute(graph, mask)
        Add an attribute of the mask indices as node attribute.
    update_graph(graph)
        Post-process the graph.
    """

    def __init__(
        self,
        nodes_name: str,
        save_mask_indices_to_attr: str | None = None,
        ignore: str | None = None,
    ) -> None:
        super().__init__(nodes_name, save_mask_indices_to_attr)
        self.ignore = ignore

    def compute_mask(self, graph: HeteroData) -> torch.Tensor:
        """Compute the mask of connected nodes."""
        nodes = graph[self.nodes_name]
        connected_mask = torch.zeros(nodes.num_nodes, dtype=torch.bool)

        if self.ignore is not None:
            LOGGER.info(f"The nodes with {self.ignore}=True will not be removed.")
            connected_mask[nodes[self.ignore].bool().squeeze()] = True

        for (source_name, _, target_name), edges in graph.edge_items():
            if source_name == self.nodes_name:
                connected_mask[edges.edge_index[0]] = True

            if target_name == self.nodes_name:
                connected_mask[edges.edge_index[1]] = True

        return connected_mask
