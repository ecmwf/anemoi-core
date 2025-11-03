# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.processors.post_process import RemoveUnconnectedNodes
from anemoi.graphs.processors.post_process import RestrictEdgeLength
from anemoi.graphs.processors.post_process import SubsetNodesInArea


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


def test_sort_edge_index_by_source_nodes(graph_nodes_and_edges: HeteroData):
    from anemoi.graphs.processors.post_process import SortEdgeIndexBySourceNodes

    processor = SortEdgeIndexBySourceNodes(descending=True)
    sorted_graph = processor.update_graph(graph_nodes_and_edges)

    expected_edge_index = torch.tensor([[3, 2, 1, 0], [2, 1, 0, 3]])

    sorted_edges = sorted_graph[("test_nodes", "to", "test_nodes")]
    assert torch.equal(sorted_edges.edge_index, expected_edge_index)
    assert torch.equal(sorted_edges.edge_attr, 10 * expected_edge_index[0][:, None])


def test_sort_edge_index_by_target_nodes(graph_nodes_and_edges: HeteroData):
    from anemoi.graphs.processors.post_process import SortEdgeIndexByTargetNodes

    processor = SortEdgeIndexByTargetNodes(descending=True)
    sorted_graph = processor.update_graph(graph_nodes_and_edges)

    expected_edge_index = torch.tensor([[0, 3, 2, 1], [3, 2, 1, 0]])

    sorted_edges = sorted_graph[("test_nodes", "to", "test_nodes")]
    assert torch.equal(sorted_edges.edge_index, expected_edge_index)
    assert torch.equal(sorted_edges.edge_attr, 10 * expected_edge_index[0][:, None])


def test_sort_edge_index_ascending_order(graph_nodes_and_edges: HeteroData):
    from anemoi.graphs.processors.post_process import SortEdgeIndexBySourceNodes

    processor = SortEdgeIndexBySourceNodes(descending=False)
    sorted_graph = processor.update_graph(graph_nodes_and_edges)

    expected_edge_index = torch.tensor([[0, 1, 2, 3], [3, 0, 1, 2]])

    sorted_edges = sorted_graph[("test_nodes", "to", "test_nodes")]
    assert torch.equal(sorted_edges.edge_index, expected_edge_index)
    assert torch.equal(sorted_edges.edge_attr, 10 * expected_edge_index[0][:, None])


def test_restrict_edge_length(graph_long_and_short_edges: HeteroData):
    """Test removal of all long ( > 1000km) edges."""
    graph = graph_long_and_short_edges
    expected_nodes_x = graph["test_nodes"].x

    short_mask = torch.tensor([1, 0, 0, 1], dtype=torch.bool)
    expected_edge_index = graph["test_nodes", "to", "test_nodes"].edge_index[:, short_mask]

    processor = RestrictEdgeLength("test_nodes", "test_nodes", 1000)
    restricted_graph = processor.update_graph(graph)

    assert torch.equal(restricted_graph["test_nodes", "to", "test_nodes"].edge_index, expected_edge_index)
    assert torch.equal(restricted_graph["test_nodes"].x, expected_nodes_x)


def test_restrict_edge_length_source_mask(graph_long_and_short_edges: HeteroData):
    """Test removal of all long ( > 1000km) edges with source in southern hemisphere."""
    graph = graph_long_and_short_edges
    expected_nodes_x = graph["test_nodes"].x

    long_southern_source_mask = torch.tensor([0, 1, 0, 0], dtype=torch.bool)
    expected_edge_index = graph["test_nodes", "to", "test_nodes"].edge_index[:, ~long_southern_source_mask]

    processor = RestrictEdgeLength("test_nodes", "test_nodes", 1000, source_mask_attr_name="southern_hemisphere_mask")
    restricted_graph = processor.update_graph(graph)

    assert torch.equal(restricted_graph["test_nodes", "to", "test_nodes"].edge_index, expected_edge_index)
    assert torch.equal(restricted_graph["test_nodes"].x, expected_nodes_x)


def test_restrict_edge_length_target_mask(graph_long_and_short_edges: HeteroData):
    """Test removal of all long ( > 1000km) edges with target in southern hemisphere."""
    graph = graph_long_and_short_edges
    expected_nodes_x = graph["test_nodes"].x

    long_southern_target_mask = torch.tensor([0, 0, 1, 0], dtype=torch.bool)
    expected_edge_index = graph["test_nodes", "to", "test_nodes"].edge_index[:, ~long_southern_target_mask]

    processor = RestrictEdgeLength("test_nodes", "test_nodes", 1000, target_mask_attr_name="southern_hemisphere_mask")
    restricted_graph = processor.update_graph(graph)

    assert torch.equal(restricted_graph["test_nodes", "to", "test_nodes"].edge_index, expected_edge_index)
    assert torch.equal(restricted_graph["test_nodes"].x, expected_nodes_x)


def test_subset_nodes_in_area(graph_long_and_short_edges: HeteroData):
    processor = SubsetNodesInArea("test_nodes", (90, -1, -90, 1))
    graph = processor.update_graph(graph_long_and_short_edges)

    # test the processor removes the nodes outside
    assert graph["test_nodes"].num_nodes == 2
    # test the processor removes the edges from/to removed nodes
    assert torch.all(graph["test_nodes", "to", "test_nodes"].edge_index == torch.tensor([[0], [1]]))
