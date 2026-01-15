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
from torch_geometric.data.storage import NodeStorage

LOGGER = logging.getLogger(__name__)


class BaseIcosahedronEdgeStrategy(ABC):
    """Abstract base class for different edge-building strategies."""

    @abstractmethod
    def get_edges(self, nodes: NodeStorage, x_hops: int, scale_resolutions: list[int]) -> NodeStorage: ...


class TriNodesEdgeBuilder(BaseIcosahedronEdgeStrategy):
    """Edge builder for TriNodes and LimitedAreaTriNodes."""

    def get_edges(
        self,
        nodes: NodeStorage,
        x_hops: int,
        scale_resolutions: list[int],
    ) -> NodeStorage:
        from anemoi.graphs.generate import tri_icosahedron

        if x_hops == 1:
            LOGGER.debug("The 1-hop edges are obtained directly from the trimesh.Trimesh object's 'edges' attribute.")
            # Compute the multiscale edges directly and store them in the node storage
            # No need of the networkx graph
            multiscale_edges = tri_icosahedron.add_1_hop_edges(
                nodes_coords_rad=nodes["x"],
                node_resolutions=nodes["_resolutions"],
                edge_resolutions=scale_resolutions,
                node_ordering=nodes["_node_ordering"],
                area_mask_builder=nodes.get("_area_mask_builder", None),
            )

        else:
            LOGGER.info("Using networkx strategy for multiscale-edge building.")
            nx_graph = tri_icosahedron.add_edges_to_nx_graph(
                nodes["_nx_graph"],
                resolutions=scale_resolutions,
                x_hops=x_hops,
                area_mask_builder=nodes.get("_area_mask_builder", None),
            )
            adjmat = nx.to_scipy_sparse_array(nx_graph, format="coo")
            # Get source & target indices of the edges
            multiscale_edges = np.stack([adjmat.col, adjmat.row], axis=0)
        return multiscale_edges


class HexNodesEdgeBuilder(BaseIcosahedronEdgeStrategy):
    """Edge builder for HexNodes and LimitedAreaHexNodes."""

    def get_edges(self, nodes: NodeStorage, x_hops: int, scale_resolutions: list[int]) -> NodeStorage:
        from anemoi.graphs.generate import hex_icosahedron

        nx_graph = hex_icosahedron.add_edges_to_nx_graph(
            nodes["_nx_graph"],
            resolutions=scale_resolutions,
            x_hops=x_hops,
        )
        adjmat = nx.to_scipy_sparse_array(nx_graph, format="coo")
        # Get source & target indices of the edges
        multiscale_edges = np.stack([adjmat.col, adjmat.row], axis=0)

        return multiscale_edges


class StretchedTriNodesEdgeBuilder(BaseIcosahedronEdgeStrategy):
    """Edge builder for StretchedTriNodes."""

    def get_edges(self, nodes: NodeStorage, x_hops: int, scale_resolutions: list[int]) -> NodeStorage:
        from anemoi.graphs.generate import tri_icosahedron
        from anemoi.graphs.generate.masks import KNNAreaMaskBuilder

        all_points_mask_builder = KNNAreaMaskBuilder("all_nodes", 1.0)
        all_points_mask_builder.fit_coords(nodes.x.numpy())

        if x_hops == 1:
            LOGGER.debug("Using tri-mesh only strategy for x_hops=1 multiscale-edge building.")
            # Compute the multiscale edges directly and store them in the node storage
            # No need of the networkx graph
            multiscale_edges = tri_icosahedron.add_1_hop_edges(
                nodes_coords_rad=nodes["x"],
                node_resolutions=nodes["_resolutions"],
                edge_resolutions=scale_resolutions,
                node_ordering=nodes["_node_ordering"],
                area_mask_builder=all_points_mask_builder,
            )

        else:
            nx_graph = tri_icosahedron.add_edges_to_nx_graph(
                nodes["_nx_graph"],
                resolutions=scale_resolutions,
                x_hops=x_hops,
                area_mask_builder=all_points_mask_builder,
            )
            adjmat = nx.to_scipy_sparse_array(nx_graph, format="coo")
            # Get source & target indices of the edges
            multiscale_edges = np.stack([adjmat.col, adjmat.row], axis=0)

        return multiscale_edges
