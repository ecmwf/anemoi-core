# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.processors.post_process import RemoveUnconnectedNodes


def test_remove_unconnected_nodes(graph_with_isolated_nodes: HeteroData):
    processor = RemoveUnconnectedNodes(nodes_name="test_nodes", ignore=None, save_mask_indices_to_attr=None)

    graph = processor.update_graph(graph_with_isolated_nodes)

    assert graph["test_nodes"].num_nodes == 4
    assert torch.equal(graph["test_nodes"].x, torch.tensor([[2], [3], [4], [5]]))
    assert "original_indices" not in graph["test_nodes"]


def test_remove_unconnected_nodes_with_indices_attr(graph_with_isolated_nodes: HeteroData):
    processor = RemoveUnconnectedNodes(
        nodes_name="test_nodes", ignore=None, save_mask_indices_to_attr="original_indices"
    )

    graph = processor.update_graph(graph_with_isolated_nodes)

    assert graph["test_nodes"].num_nodes == 4
    assert torch.equal(graph["test_nodes"].x, torch.tensor([[2], [3], [4], [5]]))
    assert torch.equal(graph["test_nodes", "to", "test_nodes"].edge_index, torch.tensor([[1, 2, 3], [0, 1, 2]]))
    assert torch.equal(graph["test_nodes"].original_indices, torch.tensor([[1], [2], [3], [4]]))


def test_remove_unconnected_nodes_with_ignore(graph_with_isolated_nodes: HeteroData):
    processor = RemoveUnconnectedNodes(nodes_name="test_nodes", ignore="mask_attr", save_mask_indices_to_attr=None)

    graph = processor.update_graph(graph_with_isolated_nodes)

    assert graph["test_nodes"].num_nodes == 5
    assert torch.equal(graph["test_nodes"].x, torch.tensor([[1], [2], [3], [4], [5]]))
    assert torch.equal(graph["test_nodes", "to", "test_nodes"].edge_index, torch.tensor([[2, 3, 4], [1, 2, 3]]))


@pytest.mark.parametrize(
    "nodes_name,ignore,save_mask_indices_to_attr",
    [
        ("test_nodes", None, "original_indices"),
        ("test_nodes", "mask_attr", None),
        ("test_nodes", None, None),
    ],
)
def test_remove_unconnected_nodes_parametrized(
    graph_with_isolated_nodes: HeteroData,
    nodes_name: str,
    ignore: str | None,
    save_mask_indices_to_attr: str | None,
):
    processor = RemoveUnconnectedNodes(
        nodes_name=nodes_name, ignore=ignore, save_mask_indices_to_attr=save_mask_indices_to_attr
    )

    graph = processor.update_graph(graph_with_isolated_nodes)

    assert isinstance(graph, HeteroData)
    pruned_nodes = 4 if ignore is None else 5
    assert graph[nodes_name].num_nodes == pruned_nodes

    if save_mask_indices_to_attr:
        assert save_mask_indices_to_attr in graph[nodes_name]
        assert graph[nodes_name][save_mask_indices_to_attr].ndim == 2
    else:
        assert graph[nodes_name].node_attrs() == graph_with_isolated_nodes[nodes_name].node_attrs()
