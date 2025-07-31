# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from abc import ABC
from abc import abstractmethod

from torch_geometric.data.storage import NodeStorage


class BaseIcosahedronEdgeStrategy(ABC):
    """Abstract base class for different edge-building strategies."""

    @abstractmethod
    def add_edges(self, nodes: NodeStorage, x_hops: int, scale_resolutions: list[int]) -> NodeStorage: ...


class TriNodesEdgeBuilder(BaseIcosahedronEdgeStrategy):
    """Edge builder for TriNodes and LimitedAreaTriNodes."""

    def add_edges(self, nodes: NodeStorage, x_hops: int, scale_resolutions: list[int]) -> NodeStorage:
        from anemoi.graphs.generate import tri_icosahedron

        nodes["_nx_graph"] = tri_icosahedron.add_edges_to_nx_graph(
            nodes["_nx_graph"],
            resolutions=scale_resolutions,
            x_hops=x_hops,
            area_mask_builder=nodes.get("_area_mask_builder", None),
        )
        return nodes


class HexNodesEdgeBuilder(BaseIcosahedronEdgeStrategy):
    """Edge builder for HexNodes and LimitedAreaHexNodes."""

    def add_edges(self, nodes: NodeStorage, x_hops: int, scale_resolutions: list[int]) -> NodeStorage:
        from anemoi.graphs.generate import hex_icosahedron

        nodes["_nx_graph"] = hex_icosahedron.add_edges_to_nx_graph(
            nodes["_nx_graph"],
            resolutions=scale_resolutions,
            x_hops=x_hops,
        )
        return nodes


class StretchedTriNodesEdgeBuilder(BaseIcosahedronEdgeStrategy):
    """Edge builder for StretchedTriNodes."""

    def add_edges(self, nodes: NodeStorage, x_hops: int, scale_resolutions: list[int]) -> NodeStorage:
        from anemoi.graphs.generate import tri_icosahedron
        from anemoi.graphs.generate.masks import KNNAreaMaskBuilder

        all_points_mask_builder = KNNAreaMaskBuilder("all_nodes", 1.0)
        all_points_mask_builder.fit_coords(nodes.x.numpy())

        nodes["_nx_graph"] = tri_icosahedron.add_edges_to_nx_graph(
            nodes["_nx_graph"],
            resolutions=scale_resolutions,
            x_hops=x_hops,
            area_mask_builder=all_points_mask_builder,
        )
        return nodes
