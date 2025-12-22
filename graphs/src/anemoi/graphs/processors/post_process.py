# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Iterable
from typing import Union

import torch
from hydra.utils import instantiate
from torch_geometric.data import HeteroData

from anemoi.graphs import EARTH_RADIUS
from anemoi.graphs.edges.attributes import EdgeLength
from anemoi.graphs.utils import NodesAxis
from anemoi.graphs.utils import get_edge_attributes

LOGGER = logging.getLogger(__name__)


class PostProcessor(ABC):
    """Base PostProcessor class."""

    @abstractmethod
    def update_graph(self, graph: HeteroData, **kwargs: Any) -> HeteroData:
        raise NotImplementedError(f"The {self.__class__.__name__} class does not implement the method update_graph().")


class BaseNodeMaskingProcessor(PostProcessor, ABC):
    """Base class for mask based node processor."""

    def __init__(
        self,
        nodes_name: Union[str, Iterable[str]],
        save_mask_indices_to_attr: str | None = None,
    ) -> None:
        self.nodes_names = (nodes_name,) if isinstance(nodes_name, str) else tuple(nodes_name)
        self.save_mask_indices_to_attr = save_mask_indices_to_attr
        super().__init__()

    def removing_nodes(self, graph: HeteroData, mask: torch.Tensor, nodes_name: str) -> HeteroData:
        """Remove nodes based on the mask passed."""
        for attr_name in graph[nodes_name].node_attrs():
            graph[nodes_name][attr_name] = graph[nodes_name][attr_name][mask]

        return graph

    def remove_edges(self, graph: HeteroData, mask: torch.Tensor, edges_name: tuple[str, str, str]) -> HeteroData:
        """Remove edges based on the mask passed."""
        for attr_name in graph[edges_name].edge_attrs():
            if attr_name == "edge_index":
                graph[edges_name][attr_name] = graph[edges_name][attr_name][:, mask]
            else:
                graph[edges_name][attr_name] = graph[edges_name][attr_name][mask]

        return graph

    def create_indices_mapper_from_mask(self, mask: torch.Tensor) -> dict[int, int]:
        return dict(zip(torch.where(mask)[0].tolist(), list(range(mask.sum()))))

    def update_edge_indices(self, graph: HeteroData, mask: torch.Tensor, nodes_name: str) -> HeteroData:
        """Update the edge indices to the new position of the nodes."""
        idx_mapping = self.create_indices_mapper_from_mask(mask)
        for edges_name in graph.edge_types:
            if edges_name[0] == nodes_name:
                valid_edges_mask = mask[graph[edges_name].edge_index[0].cpu()]
                if valid_edges_mask.any():
                    graph = self.remove_edges(graph, valid_edges_mask, edges_name)
                graph[edges_name].edge_index[0] = graph[edges_name].edge_index[0].cpu().apply_(idx_mapping.get)

            if edges_name[2] == nodes_name:
                valid_edges_mask = mask[graph[edges_name].edge_index[1].cpu()]
                if valid_edges_mask.any():
                    graph = self.remove_edges(graph, valid_edges_mask, edges_name)
                graph[edges_name].edge_index[1] = graph[edges_name].edge_index[1].cpu().apply_(idx_mapping.get)

        return graph

    @abstractmethod
    def compute_mask(self, graph: HeteroData, nodes_name: str) -> torch.Tensor: ...

    def add_attribute(self, graph: HeteroData, mask: torch.Tensor, nodes_name: str) -> HeteroData:
        """Add an attribute of the mask indices as node attribute."""
        if self.save_mask_indices_to_attr is not None:
            LOGGER.info(
                f"An attribute {self.save_mask_indices_to_attr} has been added with the indices to mask the nodes from the original graph."
            )
            mask_indices = torch.where(mask)[0].reshape((graph[nodes_name].num_nodes, -1))
            graph[nodes_name][self.save_mask_indices_to_attr] = mask_indices

        return graph

    def update_graph(self, graph: HeteroData, **kwargs: Any) -> HeteroData:
        """Post-process the graph.

        Parameters
        ----------
        graph: HeteroData
            The graph to post-process.
        kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        HeteroData
            The post-processed graph.
        """
        for nodes_name in self.nodes_names:
            mask = self.compute_mask(graph, nodes_name).cpu()
            LOGGER.info(f"Removing {(~mask).sum()} nodes from {nodes_name}.")
            graph = self.removing_nodes(graph, mask, nodes_name)
            graph = self.update_edge_indices(graph, mask, nodes_name)
            graph = self.add_attribute(graph, mask, nodes_name)
        return graph


class RemoveUnconnectedNodes(BaseNodeMaskingProcessor):
    """Remove unconnected nodes in the graph.

    Attributes
    ----------
    nodes_name: str | Iterable[str]
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
        nodes_name: Union[str, Iterable[str]],
        save_mask_indices_to_attr: str | None = None,
        ignore: str | None = None,
    ) -> None:
        super().__init__(nodes_name, save_mask_indices_to_attr)
        self.ignore = ignore

    def compute_mask(self, graph: HeteroData, nodes_name: str) -> torch.Tensor:
        """Compute the mask of connected nodes."""
        nodes = graph[nodes_name]
        connected_mask = torch.zeros(nodes.num_nodes, dtype=torch.bool)

        if self.ignore is not None:
            LOGGER.info(f"The nodes with {self.ignore}=True will not be removed.")
            connected_mask[nodes[self.ignore].bool().squeeze()] = True

        for (source_name, _, target_name), edges in graph.edge_items():
            if source_name == nodes_name:
                connected_mask[edges.edge_index[0]] = True

            if target_name == nodes_name:
                connected_mask[edges.edge_index[1]] = True

        return connected_mask


class SubsetNodesInArea(BaseNodeMaskingProcessor):
    """Remove unconnected nodes in the graph.

    Attributes
    ----------
    nodes_name : str | Iterable[str]
        Name of the unconnected nodes to remove.
    area : tuple[float, float, float, float]
        Area of interest to crop the nodes, (north, west, south, east).
    save_mask_indices_to_attr : str, optional
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
        nodes_name: Union[str, Iterable[str]],
        area: tuple[float, float, float, float],
        save_mask_indices_to_attr: str | None = None,
    ) -> None:
        super().__init__(nodes_name, save_mask_indices_to_attr)
        self.north, self.west, self.south, self.east = area

        # Convert area limits to [-90, 90]x[0,360]
        self.west, self.east = self.west % 360, (self.east - 1) % 360 + 1
        assert (self.west - self.east) % 360 != 0, "West and East limits must be different."

    def points_inside_area(
        self,
        lats: torch.Tensor,
        lons: torch.Tensor,
    ) -> torch.Tensor:
        # Convert lat & lons to [-90, 90]x[0,360)
        lons %= 360

        assert (-90 <= lats).all() and (lats <= 90).all(), "Latitude values must be in the range [-90, 90]."
        assert (0 <= lons).all() and (lons <= 360).all(), "Longitude values must be in the range [0, 360]."

        in_lat_range = (lats >= self.south) & (lats <= self.north)
        if self.west < self.east:
            in_lon_range = (lons >= self.west) & (lons <= self.east)
        else:
            # e.g: W=-10,E=10. They will be mapped to west=350, east=10.
            in_lon_range = (lons >= self.west) | (lons <= self.east)
        return in_lat_range & in_lon_range

    def compute_mask(self, graph: HeteroData, nodes_name: str) -> torch.Tensor:
        """Compute the mask of connected nodes."""
        latlons = torch.rad2deg(graph[nodes_name].x)
        mask = self.points_inside_area(lats=latlons[:, 0], lons=latlons[:, 1])
        assert mask.any(), f"The area selected does not contain any {nodes_name} node."
        return mask


class BaseSortEdgeIndex(PostProcessor, ABC):
    """Base class for sort edge indices processor."""

    nodes_axis: NodesAxis | None = None

    def __init__(self, descending: bool = True) -> None:
        assert self.nodes_axis is not None, f"{self.__class__.__name__} must define the nodes_axis class attribute."
        self.descending = descending
        super().__init__()

    def get_sorting_mask(self, edges: dict) -> torch.Tensor:
        sort_indices = torch.sort(edges["edge_index"], descending=self.descending, dim=1)
        return sort_indices.indices[self.nodes_axis.value]

    @staticmethod
    def get_edge_dim(edge_attr: str) -> int:
        # edge_index has shape (2, n_edges) and other edge attributes have (n_edges, attr_dim)
        return int(edge_attr == "edge_index")

    @staticmethod
    def sort_by_indices(x: torch.Tensor, indices: torch.Tensor, dim: int = 1) -> torch.Tensor:
        return x.index_select(dim=dim, index=indices)

    def update_graph(self, graph: HeteroData) -> HeteroData:
        """Sort all edge indices in the graph.

        Parameters
        ----------
        graph: HeteroData
            The graph to post-process.

        Returns
        -------
        HeteroData
            The post-processed graph.
        """
        for (src, to, dst), edges in graph.edge_items():
            sort_indices = self.get_sorting_mask(edges)
            for edge_attr_name in edges.edge_attrs():
                dim = BaseSortEdgeIndex.get_edge_dim(edge_attr_name)
                edge_attr = BaseSortEdgeIndex.sort_by_indices(edges[edge_attr_name], sort_indices, dim=dim)
                graph[(src, to, dst)][edge_attr_name] = edge_attr
        return graph


class SortEdgeIndexBySourceNodes(BaseSortEdgeIndex):
    nodes_axis = NodesAxis.SOURCE


class SortEdgeIndexByTargetNodes(BaseSortEdgeIndex):
    nodes_axis = NodesAxis.TARGET


class BaseEdgeMaskingProcessor(PostProcessor, ABC):
    """Base class for mask based edge processor.

    Attributes
    ----------
    source_name: str
        Name of the source nodes of edges to remove.
    target_name: str
        Name of the target nodes of edges to remove.
    """

    def __init__(
        self,
        source_name: str,
        target_name: str,
    ) -> None:
        self.source_name = source_name
        self.target_name = target_name
        self.edges_name = (self.source_name, "to", self.target_name)
        super().__init__()

    def removing_edges(self, graph: HeteroData, mask: torch.Tensor) -> HeteroData:
        """Remove edges based on the mask passed."""
        for attr_name in graph[self.edges_name].edge_attrs():
            if attr_name == "edge_index":
                graph[self.edges_name][attr_name] = graph[self.edges_name][attr_name].cpu()[:, mask]
            else:
                graph[self.edges_name][attr_name] = graph[self.edges_name][attr_name].cpu()[mask, :]

        return graph

    @abstractmethod
    def compute_mask(self, graph: HeteroData) -> torch.Tensor: ...

    def recompute_attributes(self, graph: HeteroData, graph_config: dict) -> HeteroData:
        """Recompute attributes"""
        edge_attributes = get_edge_attributes(graph_config, self.source_name, self.target_name)
        for attr_name, edge_attr_builder in edge_attributes.items():
            LOGGER.info(f"Recomputing edge attribute {attr_name}.")
            graph[self.edges_name][attr_name] = instantiate(edge_attr_builder)(
                x=(graph[self.source_name], graph[self.target_name]), edge_index=graph[self.edges_name].edge_index
            )
        return graph

    def update_graph(self, graph: HeteroData, **kwargs: Any) -> HeteroData:
        """Post-process the graph.

        Parameters
        ----------
        graph: HeteroData
            The graph to post-process.
        kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        HeteroData
            The post-processed graph.
        """
        mask = self.compute_mask(graph).cpu()
        LOGGER.info(f"Removing {(~mask).sum()} edges from {self.edges_name}.")
        graph = self.removing_edges(graph, mask)
        graph_config = kwargs.get("graph_config", {})
        graph = self.recompute_attributes(graph, graph_config)
        return graph


class RestrictEdgeLength(BaseEdgeMaskingProcessor):
    """Remove edges longer than a given treshold from the graph.

    Attributes
    ----------
    source_name: str
        Name of the source nodes of edges to remove.
    target_name: str
        Name of the target nodes of edges to remove.
    max_length_km: float
        The maximal length of edges not to be removed.
    source_mask_attr_name: str , optional
         the postprocessing will be restricted to edges with source node having True in this mask_attr
    target_mask_attr_name: str, optional
        the postprocessing will be restricted to edges with target node having True in this mask_attr
    Methods
    -------
    compute_mask(graph)
        Compute the mask of the relevant edges longer than max_length_km.
    """

    def __init__(
        self,
        source_name: str,
        target_name: str,
        max_length_km: float,
        source_mask_attr_name: str | None = None,
        target_mask_attr_name: str | None = None,
    ) -> None:
        super().__init__(source_name, target_name)
        self.treshold = max_length_km
        self.source_mask_attr_name = source_mask_attr_name
        self.target_mask_attr_name = target_mask_attr_name

    def compute_mask(self, graph: HeteroData) -> torch.Tensor:
        source_nodes = graph[self.source_name]
        target_nodes = graph[self.target_name]
        edge_index = graph[self.edges_name].edge_index
        lengths = EARTH_RADIUS * EdgeLength()(x=(source_nodes, target_nodes), edge_index=edge_index)
        mask = torch.where(lengths > self.treshold, False, True).squeeze().cpu()
        cases = [
            (self.source_mask_attr_name, source_nodes, 0),
            (self.target_mask_attr_name, target_nodes, 1),
        ]
        for mask_attr_name, nodes, i in cases:
            if mask_attr_name:
                attr_mask = nodes[mask_attr_name].squeeze()
                edge_mask = attr_mask[edge_index[i]].cpu()
                mask = torch.logical_or(mask, ~edge_mask)
        return mask
