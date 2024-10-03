from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod

import networkx as nx
import numpy as np
import torch
from anemoi.utils.config import DotDict
from torch_geometric.data import HeteroData

from anemoi.graphs.generate.hex_icosahedron import create_hex_nodes
from anemoi.graphs.generate.masks import KNNAreaMaskBuilder
from anemoi.graphs.generate.tri_icosahedron import create_tri_nodes
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder

LOGGER = logging.getLogger(__name__)


class IcosahedralNodes(BaseNodeBuilder, ABC):
    """Nodes based on iterative refinements of an icosahedron.

    Attributes
    ----------
    resolution : list[int] | int
        Refinement level of the mesh.
    name : str
        Name of the nodes.
    """

    def __init__(
        self,
        resolution: int | list[int],
        name: str,
    ) -> None:
        if isinstance(resolution, int):
            self.resolutions = list(range(resolution + 1))
        else:
            self.resolutions = resolution

        super().__init__(name)

    def get_coordinates(self) -> torch.Tensor:
        """Get the coordinates of the nodes.

        Returns
        -------
        torch.Tensor of shape (num_nodes, 2)
            A 2D tensor with the coordinates, in radians.
        """
        self.nx_graph, coords_rad, self.node_ordering = self.create_nodes()
        return torch.tensor(coords_rad[self.node_ordering], dtype=torch.float32)

    @abstractmethod
    def create_nodes(self) -> tuple[nx.DiGraph, np.ndarray, list[int]]: ...

    def register_attributes(self, graph: HeteroData, config: DotDict) -> HeteroData:
        graph[self.name]["_resolutions"] = self.resolutions
        graph[self.name]["_nx_graph"] = self.nx_graph
        graph[self.name]["_node_ordering"] = self.node_ordering
        graph[self.name]["_aoi_mask_builder"] = self.aoi_mask_builder
        return super().register_attributes(graph, config)


class LimitedAreaIcosahedralNodes(IcosahedralNodes):
    """Nodes based on iterative refinements of an icosahedron using an area of interest.

    Attributes
    ----------
    aoi_mask_builder : KNNAreaMaskBuilder
        The area of interest mask builder.
    """

    def __init__(
        self,
        resolution: int | list[int],
        reference_node_name: str,
        name: str,
        mask_attr_name: str | None = None,
        margin_radius_km: float = 100.0,
    ) -> None:

        super().__init__(resolution, name)

        self.aoi_mask_builder = KNNAreaMaskBuilder(reference_node_name, margin_radius_km, mask_attr_name)

    def register_nodes(self, graph: HeteroData) -> None:
        self.aoi_mask_builder.fit(graph)
        return super().register_nodes(graph)


class TriNodes(IcosahedralNodes):
    """Nodes based on iterative refinements of an icosahedron.

    It depends on the trimesh Python library.
    """

    def create_nodes(self) -> tuple[nx.Graph, np.ndarray, list[int]]:
        return create_tri_nodes(resolution=max(self.resolutions))


class HexNodes(IcosahedralNodes):
    """Nodes based on iterative refinements of an icosahedron.

    It depends on the h3 Python library.
    """

    def create_nodes(self) -> tuple[nx.Graph, np.ndarray, list[int]]:
        return create_hex_nodes(resolution=max(self.resolutions))


class LimitedAreaTriNodes(LimitedAreaIcosahedralNodes):
    """Nodes based on iterative refinements of an icosahedron using an area of interest.

    It depends on the trimesh Python library.

    Parameters
    ----------
    aoi_mask_builder: KNNAreaMaskBuilder
        The area of interest mask builder.
    """

    def create_nodes(self) -> tuple[nx.Graph, np.ndarray, list[int]]:
        return create_tri_nodes(resolution=max(self.resolutions), aoi_mask_builder=self.aoi_mask_builder)


class LimitedAreaHexNodes(LimitedAreaIcosahedralNodes):
    """Nodes based on iterative refinements of an icosahedron using an area of interest.

    It depends on the h3 Python library.

    Parameters
    ----------
    aoi_mask_builder: KNNAreaMaskBuilder
        The area of interest mask builder.
    """

    def create_nodes(self) -> tuple[nx.Graph, np.ndarray, list[int]]:
        return create_hex_nodes(resolution=max(self.resolutions), aoi_mask_builder=self.aoi_mask_builder)
