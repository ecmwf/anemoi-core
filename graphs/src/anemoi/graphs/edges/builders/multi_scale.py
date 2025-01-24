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

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.edges.builders.base import BaseEdgeBuilder
from anemoi.graphs.generate import hex_icosahedron
from anemoi.graphs.generate import tri_icosahedron
from anemoi.graphs.generate.masks import KNNAreaMaskBuilder
from anemoi.graphs.nodes.builders.from_refined_icosahedron import HexNodes
from anemoi.graphs.nodes.builders.from_refined_icosahedron import LimitedAreaHexNodes
from anemoi.graphs.nodes.builders.from_refined_icosahedron import LimitedAreaTriNodes
from anemoi.graphs.nodes.builders.from_refined_icosahedron import StretchedTriNodes
from anemoi.graphs.nodes.builders.from_refined_icosahedron import TriNodes
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class MultiScaleEdges(BaseEdgeBuilder):
    """Base class for multi-scale edges in the nodes of a graph.

    Attributes
    ----------
    source_name : str
        The name of the source nodes.
    target_name : str
        The name of the target nodes.
    x_hops : int
        Number of hops (in the refined icosahedron) between two nodes to connect
        them with an edge.

    Methods
    -------
    register_edges(graph)
        Register the edges in the graph.
    register_attributes(graph, config)
        Register attributes in the edges of the graph.
    update_graph(graph, attrs_config)
        Update the graph with the edges.
    """

    VALID_NODES = [
        TriNodes,
        HexNodes,
        LimitedAreaTriNodes,
        LimitedAreaHexNodes,
        StretchedTriNodes,
    ]

    def __init__(self, source_name: str, target_name: str, x_hops: int, **kwargs):
        super().__init__(source_name, target_name)
        assert source_name == target_name, f"{self.__class__.__name__} requires source and target nodes to be the same."
        assert isinstance(x_hops, int), "Number of x_hops must be an integer"
        assert x_hops > 0, "Number of x_hops must be positive"
        self.x_hops = x_hops

    def add_edges_from_tri_nodes(self, nodes: NodeStorage) -> NodeStorage:
        nodes["_nx_graph"] = tri_icosahedron.add_edges_to_nx_graph(
            nodes["_nx_graph"],
            resolutions=nodes["_resolutions"],
            x_hops=self.x_hops,
            area_mask_builder=nodes.get("_area_mask_builder", None),
        )

        return nodes

    def add_edges_from_stretched_tri_nodes(self, nodes: NodeStorage) -> NodeStorage:
        all_points_mask_builder = KNNAreaMaskBuilder("all_nodes", 1.0)
        all_points_mask_builder.fit_coords(nodes.x.numpy())

        nodes["_nx_graph"] = tri_icosahedron.add_edges_to_nx_graph(
            nodes["_nx_graph"],
            resolutions=nodes["_resolutions"],
            x_hops=self.x_hops,
            area_mask_builder=all_points_mask_builder,
        )
        return nodes

    def add_edges_from_hex_nodes(self, nodes: NodeStorage) -> NodeStorage:
        nodes["_nx_graph"] = hex_icosahedron.add_edges_to_nx_graph(
            nodes["_nx_graph"],
            resolutions=nodes["_resolutions"],
            x_hops=self.x_hops,
        )

        return nodes

    def compute_edge_index(self, source_nodes: NodeStorage, target_nodes: NodeStorage) -> torch.Tensor:
        if source_nodes.node_type in [TriNodes.__name__, LimitedAreaTriNodes.__name__]:
            source_nodes = self.add_edges_from_tri_nodes(source_nodes)
        elif source_nodes.node_type in [HexNodes.__name__, LimitedAreaHexNodes.__name__]:
            source_nodes = self.add_edges_from_hex_nodes(source_nodes)
        elif source_nodes.node_type == StretchedTriNodes.__name__:
            source_nodes = self.add_edges_from_stretched_tri_nodes(source_nodes)
        else:
            raise ValueError(f"Invalid node type {source_nodes.node_type}")

        adjmat = nx.to_scipy_sparse_array(source_nodes["_nx_graph"], format="coo")

        # Get source & target indices of the edges
        edge_index = np.stack([adjmat.col, adjmat.row], axis=0)

        return torch.from_numpy(edge_index).to(torch.int32)

    def update_graph(self, graph: HeteroData, attrs_config: DotDict | None = None) -> HeteroData:
        node_type = graph[self.source_name].node_type
        valid_node_names = [n.__name__ for n in self.VALID_NODES]
        assert node_type in valid_node_names, f"{self.__class__.__name__} requires {','.join(valid_node_names)} nodes."

        return super().update_graph(graph, attrs_config)
