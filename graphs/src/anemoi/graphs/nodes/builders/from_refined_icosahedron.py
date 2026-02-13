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

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.generate.masks import KNNAreaMaskBuilder
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder

LOGGER = logging.getLogger(__name__)


class IcosahedralNodes(BaseNodeBuilder, ABC):
    """Nodes based on iterative refinements of an icosahedron.

    Attributes
    ----------
    resolution : list[int] | int
        Refinement level of the mesh.
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
        self.hidden_attributes = BaseNodeBuilder.hidden_attributes | {
            "resolutions",
            "nx_graph",
            "node_ordering",
        }
        if not hasattr(self, "multi_scale_edge_cls"):
            raise AttributeError("Classes inheriting from IcosahedralNodes must set 'multi_scale_edge_cls' attribute.")

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


class LimitedAreaIcosahedralNodes(IcosahedralNodes, ABC):
    """Nodes based on iterative refinements of an icosahedron using an area of interest.

    Attributes
    ----------
    area_mask_builder : KNNAreaMaskBuilder
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
        self.hidden_attributes = self.hidden_attributes | {"area_mask_builder"}

        self.area_mask_builder = KNNAreaMaskBuilder(reference_node_name, margin_radius_km, mask_attr_name)

    def register_nodes(self, graph: HeteroData) -> None:
        self.area_mask_builder.fit(graph)
        return super().register_nodes(graph)


class TriNodes(IcosahedralNodes):
    """Nodes based on iterative refinements of an icosahedron.

    It depends on the trimesh Python library.
    """

    multi_scale_edge_cls: str = "anemoi.graphs.generate.multi_scale_edges.TriNodesEdgeBuilder"

    def create_nodes(self) -> tuple[nx.Graph, np.ndarray, list[int]]:
        from anemoi.graphs.generate.tri_icosahedron import create_tri_nodes

        return create_tri_nodes(resolution=max(self.resolutions))


class HexNodes(IcosahedralNodes):
    """Nodes based on iterative refinements of an icosahedron.

    It depends on the h3 Python library.
    """

    multi_scale_edge_cls: str = "anemoi.graphs.generate.multi_scale_edges.HexNodesEdgeBuilder"

    def create_nodes(self) -> tuple[nx.Graph, np.ndarray, list[int]]:
        from anemoi.graphs.generate.hex_icosahedron import create_hex_nodes

        return create_hex_nodes(resolution=max(self.resolutions))


class LimitedAreaTriNodes(LimitedAreaIcosahedralNodes):
    """Nodes based on iterative refinements of an icosahedron using an area of interest.

    It depends on the trimesh Python library.

    Parameters
    ----------
    area_mask_builder: KNNAreaMaskBuilder
        The area of interest mask builder.
    """

    multi_scale_edge_cls: str = "anemoi.graphs.generate.multi_scale_edges.TriNodesEdgeBuilder"

    def create_nodes(self) -> tuple[nx.Graph, np.ndarray, list[int]]:
        from anemoi.graphs.generate.tri_icosahedron import create_tri_nodes

        return create_tri_nodes(resolution=max(self.resolutions), area_mask_builder=self.area_mask_builder)


class LimitedAreaHexNodes(LimitedAreaIcosahedralNodes):
    """Nodes based on iterative refinements of an icosahedron using an area of interest.

    It depends on the h3 Python library.

    Parameters
    ----------
    area_mask_builder: KNNAreaMaskBuilder
        The area of interest mask builder.
    """

    multi_scale_edge_cls: str = "anemoi.graphs.generate.multi_scale_edges.HexNodesEdgeBuilder"

    def create_nodes(self) -> tuple[nx.Graph, np.ndarray, list[int]]:
        from anemoi.graphs.generate.hex_icosahedron import create_hex_nodes

        return create_hex_nodes(resolution=max(self.resolutions), area_mask_builder=self.area_mask_builder)


class StretchedIcosahedronNodes(LimitedAreaIcosahedralNodes, ABC):
    """Nodes based on iterative refinements of an icosahedron with 2
    different resolutions.

    Attributes
    ----------
    area_mask_builder : KNNAreaMaskBuilder
        The area of interest mask builder.
    """

    def __init__(
        self,
        global_resolution: int,
        lam_resolution: int,
        name: str,
        reference_node_name: str,
        mask_attr_name: str | None = None,
        margin_radius_km: float = 100.0,
    ) -> None:
        super().__init__(
            resolution=lam_resolution,
            reference_node_name=reference_node_name,
            mask_attr_name=mask_attr_name,
            margin_radius_km=margin_radius_km,
            name=name,
        )
        self.global_resolution = global_resolution


class StretchedTriNodes(StretchedIcosahedronNodes):
    """Nodes based on iterative refinements of an icosahedron with 2
    different resolutions.

    It depends on the trimesh Python library.
    """

    multi_scale_edge_cls: str = "anemoi.graphs.generate.multi_scale_edges.StretchedTriNodesEdgeBuilder"

    def create_nodes(self) -> tuple[nx.Graph, np.ndarray, list[int]]:
        from anemoi.graphs.generate.tri_icosahedron import create_stretched_tri_nodes

        return create_stretched_tri_nodes(
            base_resolution=self.global_resolution,
            lam_resolution=max(self.resolutions),
            area_mask_builder=self.area_mask_builder,
        )
