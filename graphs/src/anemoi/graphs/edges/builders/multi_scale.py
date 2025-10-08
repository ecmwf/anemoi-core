# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import importlib
import logging

import networkx as nx
import numpy as np
import torch
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.edges.builders.base import BaseEdgeBuilder
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
    scale_resolutions : int, list[int], optional
        Defines the refinement levels at which edges are computed. If an integer is provided, edges are computed for all
        levels up to and including that level. For instance, `scale_resolutions=4` includes edges at levels 1 through 4,
        whereas `scale_resolutions=[4]` only includes edges at level 4.

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

    def __init__(
        self,
        source_name: str,
        target_name: str,
        x_hops: int,
        scale_resolutions: int | list[int] | None = None,
        **kwargs,
    ):
        super().__init__(source_name, target_name)
        assert source_name == target_name, f"{self.__class__.__name__} requires source and target nodes to be the same."
        assert isinstance(x_hops, int), "Number of x_hops must be an integer"
        assert x_hops > 0, "Number of x_hops must be positive"
        self.x_hops = x_hops
        if isinstance(scale_resolutions, int):
            assert scale_resolutions > 0, "The scale_resolutions argument only supports positive integers."
            scale_resolutions = list(range(1, scale_resolutions + 1))
        assert not isinstance(scale_resolutions, str), "The scale_resolutions argument is not valid."
        assert (
            scale_resolutions is None or min(scale_resolutions) > 0
        ), "The scale_resolutions argument only supports positive integers."
        self.scale_resolutions = scale_resolutions

    @staticmethod
    def get_edge_builder_class(node_type: str) -> type[BaseIcosahedronEdgeStrategy]:
        # All node builders inheriting from IcosahedronNodes have an attribute multi_scale_edge_cls
        module = importlib.import_module("anemoi.graphs.nodes.builders.from_refined_icosahedron")
        node_cls = getattr(module, node_type, None)

        if node_cls is None:
            raise ValueError(f"Invalid node_type, {node_type}, for building multi scale edges.")

        # Instantiate the BaseIcosahedronEdgeStrategy based on the node type
        module_name = ".".join(node_cls.multi_scale_edge_cls.split(".")[:-1])
        class_name = node_cls.multi_scale_edge_cls.split(".")[-1]

        edge_builder_cls = getattr(importlib.import_module(module_name), class_name)
        return edge_builder_cls

    def compute_edge_index(self, source_nodes: NodeStorage, _target_nodes: NodeStorage) -> torch.Tensor:
        edge_builder_cls = MultiScaleEdges.get_edge_builder_class(source_nodes.node_type)

        # Get the refinement levels (or scales) at which to compute the neighbourhoods.
        scale_resolutions = self.scale_resolutions or source_nodes["_resolutions"]
        if self.scale_resolutions is not None and max(self.scale_resolutions) > max(source_nodes["_resolutions"]):
            LOGGER.warning(
                f"Some scale resolutions may be ignored because they are greater than the resolution of the nodes ({max(source_nodes['_resolutions'])})."
            )

        # Add edges
        source_nodes = edge_builder_cls().add_edges(source_nodes, self.x_hops, scale_resolutions=scale_resolutions)
        adjmat = nx.to_scipy_sparse_array(source_nodes["_nx_graph"], format="coo")

        # Get source & target indices of the edges
        edge_index = np.stack([adjmat.col, adjmat.row], axis=0)

        return torch.from_numpy(edge_index).to(torch.int32)
