# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from torch_geometric.data import HeteroData

from anemoi.graphs.create import GraphCreator


def test_graphs_python_api_smoke(mock_grids_path: tuple[str, int]) -> None:
    grids_path, _ = mock_grids_path
    config = {
        "nodes": {
            "data": {
                "node_builder": {
                    "_target_": "anemoi.graphs.nodes.NPZFileNodes",
                    "npz_file": f"{grids_path}/grid-o16.npz",
                },
            },
        },
        "edges": [
            {
                "source_name": "data",
                "target_name": "data",
                "edge_builders": [
                    {
                        "_target_": "anemoi.graphs.edges.KNNEdges",
                        "num_nearest_neighbours": 2,
                    },
                ],
                "attributes": {},
            },
        ],
    }

    graph = GraphCreator(config=config).create()

    assert isinstance(graph, HeteroData)
    assert "data" in graph.node_types
    assert ("data", "to", "data") in graph.edge_types
