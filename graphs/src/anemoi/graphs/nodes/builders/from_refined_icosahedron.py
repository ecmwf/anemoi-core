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
    """Node builder for stretched icosahedral grids supporting single resolution and multi resolution AOI configurations.

    This class allows you to define a mesh with a coarse global resolution and one or more high-resolution areas of interests (AOI).
    - In single resolution mode, you specify a single area of interest using `lam_resolution` and `reference_node_name`.
    - In multi resolution mode, you provide a list of region specifications via `lam_regions`, each with its own resolution and mask parameters.

    Parameters
    ----------
    global_resolution : int
        The base mesh resolution used outside all AOI regions.
    lam_resolution : int, optional
        (Single resolution mode only) The high-resolution refinement level for AOI.
    name : str, optional
        Name of the nodes, key for the nodes in the HeteroData graph object.
    reference_node_name : str, optional
        (Single resolution mode only) Name of the reference nodes in the graph to consider for AOI mask.
    mask_attr_name : str, optional
        (Single resolution mode only) Name of a node attribute to mask the reference nodes.
    margin_radius_km : float, optional
        (Single resolution mode only) Maximum distance to the reference nodes to consider a node as valid, in kilometers.
    lam_regions : list of dict, optional
        (Multi resolution mode only) List of AOI regions specifications. Each dict should contain keys:
        'lam_resolution', 'reference_node_name', 'mask_attr_name', 'margin_radius_km'. These keys have
        the same definitions as in single resolution mode.

    Attributes
    ----------
    global_resolution : int
        The base mesh resolution used outside all AOI regions.
    _lam_regions_raw : list[dict] or None
        Raw region specifications (multi-resolution mode), or None in single-resolution mode.
    _region_mask_builders : list[KNNAreaMaskBuilder] or None
        List of mask builders for each region (multi-resolution mode), or None in single-resolution mode.
    area_mask_builder : KNNAreaMaskBuilder
        Mask builder for AOI (single-resolution mode only, inherited from LimitedAreaIcosahedralNodes).
    """

    def __init__(
        self,
        global_resolution: int,
        lam_resolution: int | None = None,
        name: str | None = None,
        reference_node_name: str | None = None,
        mask_attr_name: str | None = None,
        margin_radius_km: float = 100.0,
        lam_regions: list[dict] | None = None,
    ) -> None:
        self._lam_regions_raw: list[dict] | None = None
        self._region_mask_builders: list[KNNAreaMaskBuilder] | None = None

        if lam_regions is not None:
            # Initialization for multi resolution mode with multi mask builders
            self._lam_regions_raw = self._parse_lam_regions(lam_regions)
            self._region_mask_builders = [
                KNNAreaMaskBuilder(r["reference_node_name"], r["margin_radius_km"], r["mask_attr_name"])
                for r in self._lam_regions_raw
            ]
            max_lam = max(r["lam_resolution"] for r in self._lam_regions_raw)
            # Call IcosahedralNodes.__init__ directly as LimitedAreaIcosahedralNodes only allows single mask builder.
            IcosahedralNodes.__init__(self, resolution=max_lam, name=name)

        else:
            # Initialization for single resolution mode
            LimitedAreaIcosahedralNodes.__init__(
                self,
                resolution=lam_resolution,
                reference_node_name=reference_node_name,
                mask_attr_name=mask_attr_name,
                margin_radius_km=margin_radius_km,
                name=name,
            )
        self.global_resolution = global_resolution

    @staticmethod
    def _parse_lam_regions(lam_regions: list[dict]) -> list[dict]:
        """Parse lam_region specifications for multi-resolution mode."""
        parsed = []
        for r in lam_regions:
            parsed.append(
                {
                    "lam_resolution": int(r["lam_resolution"]),
                    "reference_node_name": str(r["reference_node_name"]),
                    "mask_attr_name": r.get("mask_attr_name", None),
                    "margin_radius_km": float(r.get("margin_radius_km", 100.0)),
                }
            )
        return parsed

    def register_nodes(self, graph: HeteroData) -> HeteroData:
        """Register nodes, fitting one mask builder per AOI."""
        # Multi resolution mode
        if self._region_mask_builders is not None:
            for builder in self._region_mask_builders:
                builder.fit(graph)
            # Skip LimitedAreaIcosahedralNodes.register_nodes (it expects a single area_mask_builder).
            return IcosahedralNodes.register_nodes(self, graph)

        # Single resolution mode
        return super().register_nodes(graph)


class StretchedTriNodes(StretchedIcosahedronNodes):
    """Nodes based on iterative refinements of an icosahedron with one or more resolution AOI areas.

    It depends on the trimesh Python library.
    """

    multi_scale_edge_cls: str = "anemoi.graphs.generate.multi_scale_edges.StretchedTriNodesEdgeBuilder"

    def create_nodes(self) -> tuple[nx.Graph, np.ndarray, list[int]]:
        # Multi resolution mode:
        if self._region_mask_builders is not None:
            from anemoi.graphs.generate.tri_icosahedron import LamRegionSpec
            from anemoi.graphs.generate.tri_icosahedron import create_multi_stretched_tri_nodes

            specs = [
                LamRegionSpec(lam_resolution=r["lam_resolution"], area_mask_builder=b)
                for r, b in zip(self._lam_regions_raw, self._region_mask_builders)
            ]
            return create_multi_stretched_tri_nodes(
                base_resolution=self.global_resolution,
                lam_regions=specs,
            )

        # Single resolution mode:
        from anemoi.graphs.generate.tri_icosahedron import create_stretched_tri_nodes

        return create_stretched_tri_nodes(
            base_resolution=self.global_resolution,
            lam_resolution=max(self.resolutions),
            area_mask_builder=self.area_mask_builder,
        )
