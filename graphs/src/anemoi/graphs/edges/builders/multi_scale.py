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
from typing import TYPE_CHECKING, Type
import networkx as nx
import numpy as np
import torch
import importlib
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.edges.builders.base import BaseEdgeBuilder

if TYPE_CHECKING:
    from anemoi.graphs.generate.multi_scale_edges import BaseIcosahedronEdgeStrategy


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

    Note
    ----
    `MultiScaleEdges` only supports computing the edges within a set of nodes built by an `Type[IcosahedronNodes]`.
    """

    def __init__(self, source_name: str, target_name: str, x_hops: int, **kwargs):
        super().__init__(source_name, target_name)
        assert source_name == target_name, f"{self.__class__.__name__} requires source and target nodes to be the same."
        assert isinstance(x_hops, int), "Number of x_hops must be an integer"
        assert x_hops > 0, "Number of x_hops must be positive"
        self.x_hops = x_hops

    @staticmethod
    def get_edge_builder_class(node_type: str) -> Type[BaseIcosahedronEdgeStrategy]:
        # All node builders inheriting from IcosahedronNodes have an attribute multi_scale_edge_cls
        module = importlib.import_module("anemoi.graphs.nodes.builders.from_refined_icosahedron")
        edge_cls_str = getattr(module, node_type, None).multi_scale_edge_cls

        # Instantiate the BaseIcosahedronEdgeStrategy based on the node type
        module_name = ".".join(edge_cls_str.split(".")[:-1])
        class_name = edge_cls_str.split(".")[-1]

        edge_builder_cls = getattr(importlib.import_module(module_name), class_name)

        return edge_builder_cls

    def compute_edge_index(self, source_nodes: NodeStorage, _target_nodes: NodeStorage) -> torch.Tensor:
        edge_builder_cls = MultiScaleEdges.get_edge_builder_class(source_nodes.node_type)
        source_nodes = edge_builder_cls().add_edges(source_nodes, self.x_hops)
        adjmat = nx.to_scipy_sparse_array(source_nodes["_nx_graph"], format="coo")

        # Get source & target indices of the edges
        edge_index = np.stack([adjmat.col, adjmat.row], axis=0)

        return torch.from_numpy(edge_index).to(torch.int32)
