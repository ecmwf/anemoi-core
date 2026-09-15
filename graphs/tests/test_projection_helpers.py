# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.projection_helpers import fuse_participant_graphs
from anemoi.graphs.projection_helpers import graph_participants
from anemoi.graphs.projection_helpers import participant_node_name


def _graph(num_data_nodes: int) -> HeteroData:
    graph = HeteroData()
    graph["data"].x = torch.zeros(num_data_nodes, 2)
    graph["data"].num_nodes = num_data_nodes
    graph["hidden"].x = torch.zeros(2, 2)
    graph["hidden"].num_nodes = 2
    graph[("data", "to", "hidden")].edge_index = torch.zeros(2, 1, dtype=torch.int64)
    return graph


def test_graph_participants_of_fused_graph() -> None:
    fused = fuse_participant_graphs({"west": _graph(3), "north": _graph(4)})

    assert graph_participants(fused, "data") == ["west", "north"]
    assert graph_participants(fused, "hidden") == ["west", "north"]
    assert fused["data_north"].num_nodes == 4


def test_graph_participants_of_single_domain_graph_is_empty() -> None:
    assert graph_participants(_graph(3), "data") == []
    assert graph_participants(_graph(3), "hidden") == []


def test_graph_participants_ignores_unrelated_node_groups() -> None:
    fused = fuse_participant_graphs({"west": _graph(3), "north": _graph(4)})
    fused["other"].num_nodes = 1

    assert graph_participants(fused, "other") == []
    assert graph_participants(fused, "data") == ["west", "north"]


def test_participant_node_name_without_participant_is_unchanged() -> None:
    assert participant_node_name("data", None) == "data"
    assert participant_node_name("data", "west") == "data_west"
