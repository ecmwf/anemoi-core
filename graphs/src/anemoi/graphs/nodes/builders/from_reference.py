# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.nodes.builders.base import BaseNodeBuilder
from anemoi.graphs.utils import get_grid_reference_distance

LOGGER = logging.getLogger(__name__)


class ReferenceNodes(BaseNodeBuilder):
    """Nodes copied from another node set already present in the graph."""

    def __init__(self, reference_node_name: str, name: str) -> None:
        self.reference_node_name = reference_node_name
        super().__init__(name)
        self.hidden_attributes = BaseNodeBuilder.hidden_attributes | {"reference_node_name"}

    def get_coordinates(self) -> torch.Tensor:
        msg = "ReferenceNodes does not support get_coordinates without a graph context."
        raise RuntimeError(msg)

    def register_nodes(self, graph: HeteroData) -> HeteroData:
        assert (
            self.reference_node_name in graph.node_types
        ), f"Reference node '{self.reference_node_name}' must be registered before '{self.name}'."
        assert (
            "x" in graph[self.reference_node_name]
        ), f"Reference node '{self.reference_node_name}' does not have coordinates in 'x'."

        graph[self.name].x = graph[self.reference_node_name].x.clone().to(dtype=torch.float32, device=self.device)
        graph[self.name].node_type = type(self).__name__

        if graph[self.name].num_nodes >= 2:
            graph[self.name]["_grid_reference_distance"] = get_grid_reference_distance(graph[self.name].x.cpu())
        else:
            LOGGER.warning(f"{self.__class__.__name__} registered {graph[self.name].num_nodes} nodes.")

        return graph
