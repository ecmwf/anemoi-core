# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.edges import MultiScaleEdges
from anemoi.graphs.generate.transforms import cartesian_to_latlon_rad
from anemoi.graphs.nodes import AdaptiveOrographyTriNodes
from anemoi.graphs.nodes import StretchedTriNodes
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder

pytest.importorskip("trimesh")

GLOBAL_RESOLUTION = 2
LAM_RESOLUTION = 7


@pytest.fixture
def reference_graph() -> HeteroData:
    """Data nodes on a 0.1 deg grid over the synthetic mountain of ``orography_file``."""
    lat_g, lon_g = np.meshgrid(np.arange(45.0, 47.01, 0.1), np.arange(7.0, 9.01, 0.1), indexing="ij")
    graph = HeteroData()
    coords = np.deg2rad(np.stack([lat_g.ravel(), lon_g.ravel()], axis=-1))
    graph["data"].x = torch.tensor(coords, dtype=torch.float32)
    return graph


def _build(graph: HeteroData, orography_path: str, attrs: dict | None = None) -> HeteroData:
    builder = AdaptiveOrographyTriNodes(GLOBAL_RESOLUTION, LAM_RESOLUTION, "data", orography_path, "hidden")
    graph = builder.update_graph(graph, attrs or {})
    return StretchedTriNodes(GLOBAL_RESOLUTION, LAM_RESOLUTION, "base", "data").update_graph(graph, {})


def test_init():
    builder = AdaptiveOrographyTriNodes(GLOBAL_RESOLUTION, LAM_RESOLUTION, "data", "orog.nc", "hidden")
    assert isinstance(builder, BaseNodeBuilder)
    assert builder.gradient_threshold_percentile == 75.0


def test_refines_steep_terrain_only(reference_graph, orography_file):
    graph = _build(
        reference_graph, orography_file, {"lam": {"_target_": "anemoi.graphs.nodes.attributes.LimitedAreaMask"}}
    )
    hidden, base = graph["hidden"], graph["base"]

    # The stretched mesh is kept and extra nodes are added on top of it.
    num_extra = hidden.num_nodes - base.num_nodes
    assert num_extra > 0
    torch.testing.assert_close(hidden.x[: base.num_nodes], base.x, atol=1e-6, rtol=0)

    # All extra nodes are inside the limited area, near the mountain.
    extra = hidden.x[base.num_nodes :]
    assert hidden["lam"][base.num_nodes :].all()
    distance_to_peak = torch.linalg.norm(torch.rad2deg(extra) - torch.tensor([46.0, 8.0]), dim=1)
    assert distance_to_peak.max() < 1.5


def test_flat_terrain_adds_no_nodes(reference_graph, flat_orography_file):
    graph = _build(reference_graph, flat_orography_file)
    assert graph["hidden"].num_nodes == graph["base"].num_nodes


def test_nodes_lie_on_finer_icosphere(reference_graph, orography_file):
    """Every node sits on a vertex of icosphere(lam_resolution + 1), once."""
    import trimesh

    graph = _build(reference_graph, orography_file)
    hidden = graph["hidden"]
    vertices = cartesian_to_latlon_rad(trimesh.creation.icosphere(subdivisions=LAM_RESOLUTION + 1).vertices)

    ordering = np.asarray(hidden["_node_ordering"])
    assert len(np.unique(ordering)) == hidden.num_nodes
    np.testing.assert_array_equal(hidden.x.numpy(), vertices[ordering])
    assert hidden["_resolutions"] == list(range(LAM_RESOLUTION + 2))


def test_multi_scale_edges(reference_graph, orography_file):
    graph = _build(reference_graph, orography_file)
    graph = MultiScaleEdges("hidden", "hidden", x_hops=1, scale_resolutions=LAM_RESOLUTION + 1).update_graph(graph)
    edge_index = graph["hidden", "to", "hidden"].edge_index
    assert edge_index.shape[1] > 0
    assert len(torch.unique(edge_index)) == graph["hidden"].num_nodes  # no isolated nodes


def test_multi_scale_edges_reject_unsupported_nodes(reference_graph):
    with pytest.raises(ValueError, match="Invalid node_type"):
        MultiScaleEdges.get_edge_builder_class("LatLonNodes")
