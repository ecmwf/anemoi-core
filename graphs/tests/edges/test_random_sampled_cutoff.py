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

from anemoi.graphs.edges import CutOffEdges
from anemoi.graphs.edges import RandomSampledCutOffEdges
from anemoi.graphs.edges.builders.euclidean.cutoff import random_subsample_per_group

CUTOFF_KM = 30.0  # ~8 fine nodes per coarse node at the corners, ~33 inside
WIDE_CUTOFF_KM = 150.0


@pytest.fixture
def dense_to_coarse_graph() -> HeteroData:
    """Dense 0.1 deg source grid and a coarse 1 deg target grid over the same area."""

    def grid(step):
        lat_g, lon_g = np.meshgrid(np.arange(44.0, 48.01, step), np.arange(6.0, 10.01, step), indexing="ij")
        return torch.tensor(np.deg2rad(np.stack([lat_g.ravel(), lon_g.ravel()], axis=-1)), dtype=torch.float32)

    graph = HeteroData()
    graph["fine"].x = grid(0.1)
    graph["coarse"].x = grid(1.0)
    return graph


def _edges(builder, graph) -> torch.Tensor:
    return builder.update_graph(graph.clone())["fine", "to", "coarse"].edge_index


def test_random_subsample_per_group():
    groups = np.repeat(np.arange(5), [1, 3, 10, 0, 7])
    keep = random_subsample_per_group(groups, 4, np.random.default_rng(0))
    counts = np.bincount(groups[keep], minlength=5)
    np.testing.assert_array_equal(counts, np.minimum(np.bincount(groups, minlength=5), 4))
    assert np.all(np.diff(keep) > 0)
    assert len(random_subsample_per_group(np.empty(0, dtype=int), 4, np.random.default_rng(0))) == 0


def test_samples_within_radius(dense_to_coarse_graph):
    all_edges = _edges(
        CutOffEdges("fine", "coarse", cutoff_distance_km=CUTOFF_KM, max_num_neighbours=4096), dense_to_coarse_graph
    )
    sampled = _edges(
        RandomSampledCutOffEdges("fine", "coarse", 16, cutoff_distance_km=CUTOFF_KM), dense_to_coarse_graph
    )

    # Every sampled edge is a cut-off edge, and each target keeps min(16, candidates) of them.
    num_targets = dense_to_coarse_graph["coarse"].num_nodes
    all_keys = set((all_edges[0] * num_targets + all_edges[1]).tolist())
    assert set((sampled[0] * num_targets + sampled[1]).tolist()) <= all_keys
    expected = torch.clamp(torch.bincount(all_edges[1], minlength=num_targets), max=16)
    torch.testing.assert_close(torch.bincount(sampled[1], minlength=num_targets), expected)
    assert expected.min() < 16 < torch.bincount(all_edges[1]).max()  # both regimes are exercised


def test_not_only_nearest(dense_to_coarse_graph):
    """The sample reaches further than the nearest neighbours would."""
    nearest = _edges(
        CutOffEdges("fine", "coarse", cutoff_distance_km=WIDE_CUTOFF_KM, max_num_neighbours=16), dense_to_coarse_graph
    )
    sampled = _edges(
        RandomSampledCutOffEdges("fine", "coarse", 16, cutoff_distance_km=WIDE_CUTOFF_KM), dense_to_coarse_graph
    )
    x_fine, x_coarse = dense_to_coarse_graph["fine"].x, dense_to_coarse_graph["coarse"].x

    def mean_length(edges):
        return torch.linalg.norm(x_fine[edges[0]] - x_coarse[edges[1]], dim=1).mean()

    assert mean_length(sampled) > 1.5 * mean_length(nearest)


def test_seed(dense_to_coarse_graph):
    def build(seed):
        return _edges(
            RandomSampledCutOffEdges("fine", "coarse", 16, cutoff_distance_km=CUTOFF_KM, random_seed=seed),
            dense_to_coarse_graph,
        )

    assert torch.equal(build(1), build(1))
    assert not torch.equal(build(1), build(2))


@pytest.mark.parametrize("kwargs", [{"num_neighbours": 0}, {"num_neighbours": 32, "max_candidates": 16}])
def test_fail_init(kwargs):
    with pytest.raises(AssertionError):
        RandomSampledCutOffEdges("fine", "coarse", cutoff_distance_km=CUTOFF_KM, **kwargs)
