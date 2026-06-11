# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Parity tests: _TorchClusterAreaMaskBackend vs _KDTreeAreaMaskBackend.

Verifies that both mask backends produce identical boolean masks for the same
reference coordinates and query points.  Tests are skipped when torch-cluster /
pyg-lib is not installed.
"""

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs import EARTH_RADIUS
from anemoi.graphs.generate.masks import AreaMaskBuilder
from anemoi.graphs.generate.masks import _KDTreeAreaMaskBackend
from anemoi.graphs.generate.masks import _TorchClusterAreaMaskBackend
from anemoi.graphs.utils import TORCH_CLUSTER_AVAILABLE

pytestmark = pytest.mark.skipif(
    not TORCH_CLUSTER_AVAILABLE,
    reason="torch-cluster / pyg-lib not installed; skipping parity tests",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backends(coords_rad: torch.Tensor) -> tuple[_TorchClusterAreaMaskBackend, _KDTreeAreaMaskBackend]:
    """Fit both backends to the same reference coordinates and return them."""
    tc = _TorchClusterAreaMaskBackend(device="cpu")
    kd = _KDTreeAreaMaskBackend()
    tc.fit(coords_rad)
    kd.fit(coords_rad)
    return tc, kd


# ---------------------------------------------------------------------------
# Direct backend parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("margin_radius_km", [50, 100, 500])
def test_backends_agree_on_reference_coords(margin_radius_km, graph_with_nodes: HeteroData):
    """Both backends must agree when queried with the reference coordinates.

    Every reference point should be within the margin of itself, so the mask
    should be all-True regardless of backend.
    """
    coords_rad = graph_with_nodes["test_nodes"].x
    chord_threshold = float(2 * np.sin(margin_radius_km / (2 * EARTH_RADIUS)))

    tc, kd = _make_backends(coords_rad)

    tc_mask = tc.get_mask(coords_rad, chord_threshold)
    kd_mask = kd.get_mask(coords_rad, chord_threshold)

    assert tc_mask.dtype == torch.bool
    assert kd_mask.dtype == torch.bool
    assert tc_mask.shape == kd_mask.shape
    assert torch.equal(tc_mask, kd_mask), (
        f"margin={margin_radius_km} km: " f"TC={tc_mask.tolist()}, KD={kd_mask.tolist()}"
    )


def test_backends_agree_on_numpy_query(graph_with_nodes: HeteroData):
    """Both backends must agree when the query is passed as a numpy array."""
    coords_rad = graph_with_nodes["test_nodes"].x
    chord_threshold = float(2 * np.sin(100.0 / (2 * EARTH_RADIUS)))

    tc, kd = _make_backends(coords_rad)

    query_np = coords_rad.cpu().numpy()
    tc_mask = tc.get_mask(query_np, chord_threshold)
    kd_mask = kd.get_mask(query_np, chord_threshold)

    assert torch.equal(tc_mask, kd_mask)


def test_backends_agree_distant_query(graph_with_nodes: HeteroData):
    """With a tiny radius and far-away query points, masks should both be all-False."""
    coords_rad = graph_with_nodes["test_nodes"].x
    # 1 km radius – tight enough that distant nodes are excluded
    chord_threshold = float(2 * np.sin(1.0 / (2 * EARTH_RADIUS)))

    tc, kd = _make_backends(coords_rad)

    # Query at the north pole (far from any node in the fixture)
    north_pole = torch.tensor([[(float(torch.pi) / 2), 0.0]])
    tc_mask = tc.get_mask(north_pole, chord_threshold)
    kd_mask = kd.get_mask(north_pole, chord_threshold)

    assert torch.equal(tc_mask, kd_mask)
    assert not tc_mask.any(), "Expected no matches near north pole with 1 km radius"


@pytest.mark.parametrize("margin_radius_km", [50, 200, 1000])
def test_backends_agree_partial_overlap(margin_radius_km):
    """Both backends agree on a mix of inside/outside query points."""
    rng = np.random.default_rng(42)
    # Reference: 20 random points on the unit sphere (lat/lon in radians)
    ref_lats = rng.uniform(-np.pi / 2, np.pi / 2, 20)
    ref_lons = rng.uniform(0, 2 * np.pi, 20)
    ref_coords = torch.from_numpy(np.stack([ref_lats, ref_lons], axis=1)).float()

    # Query: same points + 20 extra random ones
    q_lats = rng.uniform(-np.pi / 2, np.pi / 2, 20)
    q_lons = rng.uniform(0, 2 * np.pi, 20)
    query_coords = torch.from_numpy(
        np.concatenate([np.stack([ref_lats, ref_lons], axis=1), np.stack([q_lats, q_lons], axis=1)], axis=0)
    ).float()

    chord_threshold = float(2 * np.sin(margin_radius_km / (2 * EARTH_RADIUS)))

    tc, kd = _make_backends(ref_coords)

    tc_mask = tc.get_mask(query_coords, chord_threshold)
    kd_mask = kd.get_mask(query_coords, chord_threshold)

    assert tc_mask.dtype == torch.bool
    assert tc_mask.shape == kd_mask.shape
    assert torch.equal(tc_mask, kd_mask), (
        f"margin={margin_radius_km} km: " f"TC={tc_mask.tolist()}, KD={kd_mask.tolist()}"
    )


# ---------------------------------------------------------------------------
# High-level AreaMaskBuilder parity (patches TORCH_CLUSTER_AVAILABLE)
# ---------------------------------------------------------------------------


def test_area_mask_builder_full_pipeline_agrees(graph_with_nodes: HeteroData):
    """AreaMaskBuilder with each backend must produce the same mask.

    Patches the TORCH_CLUSTER_AVAILABLE flag to exercise both branches of the
    AreaMaskBuilder constructor.
    """
    import anemoi.graphs.generate.masks as _masks_module
    import anemoi.graphs.utils as _utils_module

    query_coords = graph_with_nodes["test_nodes"].x.cpu().numpy()

    # Force torch-cluster backend
    _masks_module.TORCH_CLUSTER_AVAILABLE = True
    _utils_module.TORCH_CLUSTER_AVAILABLE = True
    builder_tc = AreaMaskBuilder("test_nodes", margin_radius_km=100)
    builder_tc.fit(graph_with_nodes)
    mask_tc = builder_tc.get_mask(query_coords)

    # Force KDTree backend
    _masks_module.TORCH_CLUSTER_AVAILABLE = False
    _utils_module.TORCH_CLUSTER_AVAILABLE = False
    builder_kd = AreaMaskBuilder("test_nodes", margin_radius_km=100)
    builder_kd.fit(graph_with_nodes)
    mask_kd = builder_kd.get_mask(query_coords)

    assert torch.equal(mask_tc, mask_kd), f"AreaMaskBuilder full pipeline: TC={mask_tc.tolist()}, KD={mask_kd.tolist()}"
