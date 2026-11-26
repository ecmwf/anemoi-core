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

from anemoi.graphs import EARTH_RADIUS
from anemoi.graphs.generate.utils import compute_field_gradient


def _scattered_coords(num_points: int = 4000, seed: int = 0) -> np.ndarray:
    """Irregular points around (46N, 8E), in radians."""
    rng = np.random.default_rng(seed)
    return np.deg2rad(np.column_stack([rng.uniform(44, 48, num_points), rng.uniform(6, 10, num_points)]))


QUERY = np.deg2rad([[45.5, 7.5], [46.0, 8.0], [46.5, 8.5]])


def test_northward_ramp():
    """A field rising 2 units per km to the north has a gradient of 2 per km."""
    coords = _scattered_coords()
    ramp = 2.0 * EARTH_RADIUS * coords[:, 0]
    np.testing.assert_allclose(compute_field_gradient(coords, ramp, QUERY), 2.0, rtol=1e-2)


def test_eastward_ramp_accounts_for_latitude():
    coords = _scattered_coords()
    ramp = EARTH_RADIUS * np.cos(np.deg2rad(46.0)) * coords[:, 1]
    gradient = compute_field_gradient(coords, ramp, np.deg2rad([[46.0, 8.0]]))
    np.testing.assert_allclose(gradient, 1.0, rtol=1e-2)


def test_constant_field_is_exactly_zero():
    coords = _scattered_coords()
    np.testing.assert_array_equal(compute_field_gradient(coords, np.full(len(coords), 123.456), QUERY), 0.0)


def test_scales_with_the_field():
    coords = _scattered_coords()
    field = 100.0 * np.sin(20 * coords[:, 0]) * np.cos(20 * coords[:, 1])
    np.testing.assert_allclose(
        compute_field_gradient(coords, 9.80665 * field, QUERY), 9.80665 * compute_field_gradient(coords, field, QUERY)
    )


def test_missing_values_are_ignored():
    coords = _scattered_coords()
    ramp = EARTH_RADIUS * coords[:, 0]
    ramp[::3] = np.nan
    np.testing.assert_allclose(compute_field_gradient(coords, ramp, QUERY), 1.0, rtol=1e-2)


def test_all_missing_fails():
    coords = _scattered_coords(10)
    with pytest.raises(AssertionError, match="valid values"):
        compute_field_gradient(coords, np.full(10, np.nan), QUERY)


def test_no_gradient_outside_the_source_points():
    coords = _scattered_coords()
    ramp = EARTH_RADIUS * coords[:, 0]
    # On the edge of the data the fit is still valid; a few degrees away it is an extrapolation.
    gradient = compute_field_gradient(coords, ramp, np.deg2rad([[46.0, 6.0], [46.0, 4.0], [50.0, 8.0], [0.0, 0.0]]))
    assert gradient[0] == pytest.approx(1.0, rel=1e-2)
    np.testing.assert_array_equal(gradient[1:], 0.0)


def test_chunking_does_not_change_the_result():
    coords = _scattered_coords()
    field = np.sin(20 * coords[:, 0])
    np.testing.assert_array_equal(
        compute_field_gradient(coords, field, QUERY, chunk_size=1), compute_field_gradient(coords, field, QUERY)
    )
