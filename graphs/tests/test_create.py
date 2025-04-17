# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.create import GraphCreator


class TestGraphCreator:

    @pytest.mark.parametrize("name", ["graph.pt", None])
    def test_generate_graph(self, config_file: tuple[Path, str], mock_grids_path: tuple[str, int], name: str):
        """Test GraphCreator workflow."""
        tmp_path, config_name = config_file
        graph_path = tmp_path / name if isinstance(name, str) else None
        config_path = tmp_path / config_name

        graph = GraphCreator(config=config_path).create(save_path=graph_path)

        assert isinstance(graph, HeteroData)
        assert "test_nodes" in graph.node_types
        assert ("test_nodes", "to", "test_nodes") in graph.edge_types

        for nodes in graph.node_stores:
            for node_attr in nodes.node_attrs():
                assert isinstance(nodes[node_attr], torch.Tensor)
                assert nodes[node_attr].dtype in [torch.int32, torch.float32], f"{node_attr} is not int32/float32."

        for edges in graph.edge_stores:
            for edge_attr in edges.edge_attrs():
                assert isinstance(edges[edge_attr], torch.Tensor)
                assert edges[edge_attr].dtype in [torch.int32, torch.float32], f"{node_attr} is not int32/float32."

        for nodes in graph.node_stores:
            for node_attr in nodes.node_attrs():
                assert not node_attr.startswith("_")
        for edges in graph.edge_stores:
            for edge_attr in edges.edge_attrs():
                assert not edge_attr.startswith("_")

        if graph_path is not None:
            assert graph_path.exists()
            graph_saved = torch.load(graph_path, weights_only=False)
            assert graph.node_types == graph_saved.node_types
            assert graph.edge_types == graph_saved.edge_types
