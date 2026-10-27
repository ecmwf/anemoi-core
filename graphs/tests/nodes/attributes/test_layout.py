# (C) Copyright 2026 Anemoi contributors.
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

from anemoi.graphs.nodes import LatLonNodes
from anemoi.graphs.nodes import LimitedAreaRectilinearNodes
from anemoi.graphs.nodes import RectilinearNodes
from anemoi.graphs.processors.post_process import RemoveUnconnectedNodes

LAYOUT_INDEX = {"layout_index": {"_target_": "anemoi.graphs.nodes.attributes.RectilinearLayoutIndex"}}


def _reference_graph() -> HeteroData:
    return LatLonNodes([40.0, 50.0, 45.0], [0.0, 10.0, 5.0], "reference").register_nodes(HeteroData())


@pytest.mark.parametrize(
    "node_builder",
    [
        RectilinearNodes(30.0, "test_nodes"),
        LimitedAreaRectilinearNodes(1.0, "reference", "test_nodes", margin_radius_km=300.0, crop_to_margin=True),
    ],
)
def test_layout_index(node_builder):
    graph = node_builder.update_graph(_reference_graph(), LAYOUT_INDEX)
    index = graph["test_nodes"]["layout_index"]

    assert index.dtype == torch.int64
    assert index.shape == (graph["test_nodes"].num_nodes, 2)
    assert torch.equal(index, graph["test_nodes"]["_layout_index"])
    num_rows, num_cols = graph["test_nodes"]["_layout_shape"]
    assert (index >= 0).all() and (index[:, 0] < num_rows).all() and (index[:, 1] < num_cols).all()
    # Each node has its own grid point.
    assert len(torch.unique(index[:, 0] * num_cols + index[:, 1])) == len(index)


def test_layout_index_fails_for_other_nodes():
    with pytest.raises(AssertionError, match="RectilinearNodes"):
        LatLonNodes([0.0], [0.0], "test_nodes").update_graph(HeteroData(), LAYOUT_INDEX)


def test_layout_index_after_removing_nodes():
    graph = RectilinearNodes(30.0, "test_nodes").update_graph(HeteroData(), LAYOUT_INDEX)
    connected = torch.tensor([3, 14, 15, 40])
    graph["test_nodes", "to", "test_nodes"].edge_index = torch.stack([connected, connected.roll(1)])

    graph = RemoveUnconnectedNodes(nodes_name="test_nodes", ignore=None, save_mask_indices_to_attr=None).update_graph(
        graph
    )

    expected = torch.stack([connected // 12, connected % 12], dim=-1)
    assert torch.equal(graph["test_nodes"]["layout_index"].cpu(), expected)
    assert torch.equal(graph["test_nodes"]["_layout_index"].cpu(), expected)
    assert graph["test_nodes"]["_layout_shape"] == [6, 12]
