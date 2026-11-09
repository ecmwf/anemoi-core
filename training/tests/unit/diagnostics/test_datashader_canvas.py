# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np

from anemoi.training.diagnostics.evaluation.plotting.sample import _datashader_canvas_size


def test_large_dataset_keeps_square_canvas() -> None:
    # o96 (40320 points): unchanged 0.004 * n square canvas
    assert _datashader_canvas_size(40320, np.radians([0.0, 359.0]), np.radians([-89.0, 89.0])) == (161, 161)
    assert _datashader_canvas_size(10**6, np.array([0.0, 1.0]), np.array([0.0, 1.0])) == (500, 500)


def test_small_limited_area_dataset_follows_aspect_and_density() -> None:
    # 3824-point o2560 Switzerland subset: was clamped to 25x25
    width, height = _datashader_canvas_size(3824, np.radians([5.9, 10.55]), np.radians([45.75, 47.83]))
    assert (width, height) == (65, 29)
    assert 1.5 <= 3824 / (width * height) <= 2.5


def test_small_dataset_respects_minimum_side_and_degenerate_extent() -> None:
    assert _datashader_canvas_size(40, np.array([0.0, 1.0]), np.array([0.0, 1.0])) == (10, 10)
    assert _datashader_canvas_size(40, np.array([0.0, 0.0]), np.array([0.0, 1.0])) == (10, 10)


def test_scatter_marker_size_scales_with_density_and_sparsity() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    from anemoi.training.diagnostics.evaluation.plotting.sample import _scatter_marker_size

    fig, ax = plt.subplots(figsize=(4, 4))
    dense_global = _scatter_marker_size(fig, ax, np.zeros(40320))
    dense_regional = _scatter_marker_size(fig, ax, np.zeros(3824))
    sparse = np.full(3824, np.nan)
    sparse[:40] = 1.0
    sparse_regional = _scatter_marker_size(fig, ax, sparse)
    assert dense_global >= 1.0
    assert dense_regional > dense_global
    # sparse markers grow, but at most 3 cells wide (9x the area)
    assert np.isclose(sparse_regional, 9 * dense_regional)
    assert _scatter_marker_size(fig, ax, np.zeros(10**8)) == 1.0
    plt.close(fig)
