# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import logging

import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.nodes.builders.base import BaseNodeBuilder

LOGGER = logging.getLogger(__name__)


class UnionMeshNodes(BaseNodeBuilder):
    """Create a 'mesh' node set by concatenating coordinates from multiple node sets.

    This is a lightweight way to build an output grid that contains *all* observation
    points across multiple datasets.

    The builder also stores the slices used for each source node type in
    `graph[mesh]._slices` as a dict-like string for debugging.

    Parameters
    ----------
    source_names : list[str]
        Node types to concatenate, in order.
    """

    def __init__(self, *, name: str, source_names: list[str]):
        super().__init__(name=name)
        if not source_names:
            raise ValueError("source_names must be a non-empty list")
        self.source_names = list(source_names)

    def register_nodes(self, graph: HeteroData) -> HeteroData:
        xs = []
        slices = {}
        start = 0
        for s in self.source_names:
            if s not in graph.node_types or "x" not in graph[s]:
                raise KeyError(f"UnionMeshNodes: missing node type '{s}' or its 'x' coordinates")
            x = graph[s].x
            n = int(x.shape[0])
            xs.append(x)
            slices[s] = (start, start + n)
            start += n

        graph[self.name].x = torch.cat(xs, dim=0)

        # Optional bookkeeping tensors
        src_id = []
        for i, s in enumerate(self.source_names):
            a, b = slices[s]
            src_id.append(torch.full((b - a,), i, dtype=torch.int16))
        graph[self.name].source_id = torch.cat(src_id, dim=0)

        graph[self.name]._builder = self.__class__.__name__
        graph[self.name]._source_names = "|".join(self.source_names)
        graph[self.name]._slices = str(slices)

        LOGGER.info("Created union mesh '%s' with %d nodes from %s", self.name, start, self.source_names)
        return graph
