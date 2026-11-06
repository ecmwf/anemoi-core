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

from anemoi.graphs import EARTH_RADIUS
from anemoi.graphs.edges import MultiScaleEdges
from anemoi.graphs.generate.transforms import cartesian_to_latlon_rad
from anemoi.graphs.nodes import AdaptiveStretchedTriNodes
from anemoi.graphs.nodes import ProbabilisticNodes
from anemoi.graphs.nodes import ProbabilisticStretchedTriNodes
from anemoi.graphs.nodes import StretchedTriNodes
from anemoi.graphs.nodes.builders import from_adaptive
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder

pytest.importorskip("trimesh")

GLOBAL_RESOLUTION = 2
LAM_RESOLUTION = 7


@pytest.fixture
def reference_graph() -> HeteroData:
    """Data nodes on a 0.1 deg grid with three attributes.

    ``z`` is a 3000 m Gaussian mountain at (46N, 8E), ``other`` the same mountain at
    (45.5N, 7.5E), and ``flat`` a constant.
    """
    lat_g, lon_g = np.meshgrid(np.arange(45.0, 47.01, 0.1), np.arange(7.0, 9.01, 0.1), indexing="ij")
    lat, lon = lat_g.ravel(), lon_g.ravel()

    def mountain(lat0, lon0):
        values = 3000.0 * np.exp(-((lat - lat0) ** 2 + (lon - lon0) ** 2) / (2 * 0.4**2))
        return torch.tensor(values, dtype=torch.float32).unsqueeze(-1)

    graph = HeteroData()
    graph["data"].x = torch.tensor(np.deg2rad(np.stack([lat, lon], axis=-1)), dtype=torch.float32)
    graph["data"]["z"] = mountain(46.0, 8.0)
    graph["data"]["other"] = mountain(45.5, 7.5)
    graph["data"]["flat"] = torch.full((len(lat), 1), 500.0)
    return graph


def _build(graph: HeteroData, attr_name: str = "z", attrs: dict | None = None) -> HeteroData:
    builder = AdaptiveStretchedTriNodes(GLOBAL_RESOLUTION, LAM_RESOLUTION, "data", attr_name, "hidden")
    graph = builder.update_graph(graph, attrs or {})
    return StretchedTriNodes(GLOBAL_RESOLUTION, LAM_RESOLUTION, "base", "data").update_graph(graph, {})


def test_init():
    builder = AdaptiveStretchedTriNodes(GLOBAL_RESOLUTION, LAM_RESOLUTION, "data", "z", "hidden")
    assert isinstance(builder, BaseNodeBuilder)
    assert builder.gradient_threshold_percentile == 75.0


def test_requires_the_attribute_on_the_reference_nodes(reference_graph):
    with pytest.raises(AssertionError, match="missing"):
        _build(reference_graph, "missing")


def test_attribute_is_chosen_in_the_config(reference_graph):
    """Another attribute moves the extra nodes to where that attribute has large gradients."""

    def mean_distance(attr_name, peak):
        graph = _build(reference_graph.clone(), attr_name)
        extra = torch.rad2deg(graph["hidden"].x[graph["base"].num_nodes :])
        assert len(extra) > 0
        return torch.linalg.norm(extra - torch.tensor(peak), dim=1).mean()

    z_peak, other_peak = [46.0, 8.0], [45.5, 7.5]
    assert mean_distance("z", z_peak) < mean_distance("other", z_peak)
    assert mean_distance("other", other_peak) < mean_distance("z", other_peak)


def test_refines_large_gradients_only(reference_graph):
    graph = _build(reference_graph, attrs={"lam": {"_target_": "anemoi.graphs.nodes.attributes.LimitedAreaMask"}})
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


def test_constant_field_adds_no_nodes(reference_graph):
    graph = _build(reference_graph, "flat")
    assert graph["hidden"].num_nodes == graph["base"].num_nodes


def test_nodes_lie_on_finer_icosphere(reference_graph):
    """Every node sits on a vertex of icosphere(lam_resolution + 1), once."""
    import trimesh

    graph = _build(reference_graph)
    hidden = graph["hidden"]
    vertices = cartesian_to_latlon_rad(trimesh.creation.icosphere(subdivisions=LAM_RESOLUTION + 1).vertices)

    ordering = np.asarray(hidden["_node_ordering"])
    assert len(np.unique(ordering)) == hidden.num_nodes
    np.testing.assert_array_equal(hidden.x.numpy(), vertices[ordering])
    assert hidden["_resolutions"] == list(range(LAM_RESOLUTION + 2))


def test_multi_scale_edges(reference_graph):
    graph = _build(reference_graph)
    graph = MultiScaleEdges("hidden", "hidden", x_hops=1, scale_resolutions=LAM_RESOLUTION + 1).update_graph(graph)
    edge_index = graph["hidden", "to", "hidden"].edge_index
    assert edge_index.shape[1] > 0
    assert len(torch.unique(edge_index)) == graph["hidden"].num_nodes  # no isolated nodes


def test_multi_scale_edges_reject_unsupported_nodes(reference_graph):
    with pytest.raises(ValueError, match="Invalid node_type"):
        MultiScaleEdges.get_edge_builder_class("LatLonNodes")


def _min_pairwise_distance_rad(coords: np.ndarray) -> float:
    from sklearn.neighbors import BallTree

    dist, _ = BallTree(coords, metric="haversine").query(coords, k=2)
    return float(dist[:, 1].min())


def test_poisson_disk_sample_keeps_min_distance(monkeypatch):
    """The minimum distance also holds between nodes accepted between two tree rebuilds."""
    monkeypatch.setattr(from_adaptive, "_TREE_REBUILD_INTERVAL", 7)
    rng = np.random.default_rng(0)
    candidates = np.deg2rad(np.column_stack([rng.uniform(45, 47, 5000), rng.uniform(7, 9, 5000)]))
    min_dist = np.deg2rad(0.05)

    accepted = from_adaptive._poisson_disk_sample(candidates, np.empty((0, 2)), 300, min_dist)
    assert 7 < len(accepted) <= 300
    assert _min_pairwise_distance_rad(accepted.astype(float)) > min_dist * (1 - 1e-5)


class TestProbabilisticStretchedTriNodes:

    def _build(self, graph, attr_name="z"):
        builder = ProbabilisticStretchedTriNodes(
            GLOBAL_RESOLUTION, LAM_RESOLUTION, "data", attr_name, "hidden", extra_nodes_percent=50
        )
        graph = builder.update_graph(graph, {})
        return StretchedTriNodes(GLOBAL_RESOLUTION, LAM_RESOLUTION, "base", "data").update_graph(graph, {})

    def test_adds_nodes_on_large_gradients(self, reference_graph):
        graph = self._build(reference_graph)
        hidden, base = graph["hidden"], graph["base"]
        torch.testing.assert_close(hidden.x[: base.num_nodes], base.x, atol=1e-6, rtol=0)

        extra = hidden.x[base.num_nodes :]
        assert 0 < len(extra) <= 0.5 * base.num_nodes
        assert torch.linalg.norm(torch.rad2deg(extra) - torch.tensor([46.0, 8.0]), dim=1).max() < 1.5
        min_dist = 0.5 * np.sqrt(4 * np.pi / (10 * 4**LAM_RESOLUTION))
        assert _min_pairwise_distance_rad(hidden.x.double().numpy()) > min_dist * (1 - 1e-4)

    def test_constant_field_adds_no_nodes(self, reference_graph):
        graph = self._build(reference_graph, "flat")
        assert graph["hidden"].num_nodes == graph["base"].num_nodes


class TestProbabilisticNodes:

    def _build(self, graph, attr_name="z", compression_lam=0.1, seed=42):
        builder = ProbabilisticNodes(
            "hidden",
            attr_name,
            "data",
            num_nodes_global=200,
            num_nodes_lam=150,
            compression_global=10.0,
            compression_lam=compression_lam,
            min_dist_km=5.0,
            random_seed=seed,
        )
        return builder.update_graph(graph.clone(), {})["hidden"].x

    def test_counts_and_spacing(self, reference_graph):
        x = self._build(reference_graph)
        assert len(x) == 350
        assert _min_pairwise_distance_rad(x.double().numpy()) > 5.0 / EARTH_RADIUS * (1 - 1e-4)

    def test_follows_the_gradient(self, reference_graph):
        def median_distance_to_peak(x):
            lam = torch.rad2deg(x[:150])
            return torch.linalg.norm(lam - torch.tensor([46.0, 8.0]), dim=1).median()

        steep = self._build(reference_graph, compression_lam=0.1)
        uniform = self._build(reference_graph, compression_lam=10.0)
        assert median_distance_to_peak(steep) < 0.7 * median_distance_to_peak(uniform)

    def test_does_not_depend_on_the_units(self, reference_graph):
        reference = self._build(reference_graph)
        reference_graph["data"]["z"] = 1e-6 * reference_graph["data"]["z"]
        torch.testing.assert_close(self._build(reference_graph), reference)

    def test_seed(self, reference_graph):
        assert torch.equal(self._build(reference_graph), self._build(reference_graph))
        assert not torch.equal(self._build(reference_graph, seed=1), self._build(reference_graph, seed=2))
