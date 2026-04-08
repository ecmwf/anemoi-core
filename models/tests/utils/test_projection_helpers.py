# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from torch_geometric.data import HeteroData

from anemoi.graphs.projection_helpers import get_graph_node_names
from anemoi.graphs.projection_helpers import uses_fused_dataset_graph


@pytest.fixture
def fused_graph() -> HeteroData:
    graph = HeteroData()
    for node_name in ["era5", "cerra", "hidden"]:
        graph[node_name].num_nodes = 1
    return graph


def test_get_graph_node_names_and_fused_detection_work_for_graphs_and_configs(fused_graph: HeteroData) -> None:
    graph_config = {
        "nodes": {
            "era5": {},
            "cerra": {},
            "hidden": {},
        },
    }
    single_named_dataset_graph = {
        "nodes": {
            "era5": {},
            "hidden": {},
        },
    }
    single_generic_dataset_graph = {
        "nodes": {
            "data": {},
            "hidden": {},
        },
    }

    assert get_graph_node_names(fused_graph) >= {"era5", "cerra", "hidden"}
    assert get_graph_node_names(graph_config) == {"era5", "cerra", "hidden"}
    assert uses_fused_dataset_graph(fused_graph, ["era5", "cerra"]) is True
    assert uses_fused_dataset_graph(graph_config, ["era5", "cerra"]) is True
    assert uses_fused_dataset_graph(single_named_dataset_graph, ["era5"]) is True
    assert uses_fused_dataset_graph(single_generic_dataset_graph, ["era5"]) is False
