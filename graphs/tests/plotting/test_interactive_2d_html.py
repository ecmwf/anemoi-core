# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import torch

from anemoi.graphs.plotting.interactive_2d_html import plot_edges_2d


def test_plot_edges_2d_builds_figure_without_showing() -> None:
    source_coords = torch.tensor([[0.0, 0.0], [np.pi / 4, np.pi / 2]])
    target_coords = torch.tensor([[np.pi / 2, np.pi], [-np.pi / 4, 3 * np.pi / 2]])
    edge_index = torch.tensor([[0, 1], [1, 0]])

    fig = plot_edges_2d(source_coords, target_coords, edge_index, source_name="data", target_name="hidden", show=False)

    assert len(fig.data) == 3
    assert fig.data[0].mode == "lines"
    assert fig.data[0].name == "Connections"
    assert fig.data[1].name == "data"
    assert fig.data[2].name == "hidden"
    np.testing.assert_allclose(np.array(fig.data[0].lat), np.array([0.0, -45.0, np.nan, 45.0, 90.0, np.nan]))


def test_plot_edges_2d_can_hide_node_traces() -> None:
    source_coords = torch.zeros((1, 2))
    target_coords = torch.zeros((1, 2))
    edge_index = torch.tensor([[0], [0]])

    fig = plot_edges_2d(source_coords, target_coords, edge_index, show_nodes=False, show=False)

    assert len(fig.data) == 1
