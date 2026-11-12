# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Graph stand-ins for model stubs built without ``__init__``."""

from __future__ import annotations

from torch_geometric.data import HeteroData

from anemoi.models.layers.graph import StaticGraphGeometry


def stub_graph_geometry(**num_nodes: int) -> StaticGraphGeometry:
    """A :class:`StaticGraphGeometry` holding only node counts, ``{node_set: num_nodes}``."""
    graph = HeteroData()
    for name, count in num_nodes.items():
        graph[name].num_nodes = count
    return StaticGraphGeometry(graph, coordinate_node_names=[])
