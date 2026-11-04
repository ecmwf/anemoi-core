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

from anemoi.graphs.generate.utils import compute_orography_elevation
from anemoi.graphs.generate.utils import compute_orography_gradient
from anemoi.graphs.generate.utils import load_orography_grid
from tests.conftest import _write_orography

EARTH_RADIUS_M = 6_371_000.0
LATS = np.arange(44.0, 48.0001, 0.05)
LONS = np.arange(6.0, 10.0001, 0.05)


def _rad(lat_lon_deg) -> np.ndarray:
    return np.deg2rad(np.asarray(lat_lon_deg, dtype=float))


def test_load_orography_grid(orography_file):
    lats, lons, orog = load_orography_grid(orography_file)
    assert orog.shape == (len(lats), len(lons))
    assert np.all(np.diff(lats) > 0)
    assert orog.max() == pytest.approx(3000.0, rel=1e-6)


def test_load_orography_grid_flips_descending_latitudes(tmp_path):
    lat_g, _ = np.meshgrid(LATS, LONS, indexing="ij")
    path = _write_orography(tmp_path / "desc.nc", LATS[::-1], LONS, lat_g[::-1])
    lats, _, orog = load_orography_grid(path)
    assert np.all(np.diff(lats) > 0)
    np.testing.assert_allclose(orog[:, 0], lats)


@pytest.mark.parametrize(
    "name, units, height_m, converted",
    [
        ("z", None, 500.0, True),  # geopotential name, low terrain
        ("orog", "m**2 s**-2", 500.0, True),  # geopotential units
        ("orog", "m", 4500.0, False),  # metres
        ("z", "m", 500.0, False),  # units win over the name
        ("hsurf", None, 1500.0, True),  # no metadata: magnitude above 10 000
        ("hsurf", None, 500.0, False),
    ],
)
def test_load_orography_grid_geopotential(tmp_path, name, units, height_m, converted):
    import xarray as xr

    values = np.full((len(LATS), len(LONS)), height_m * 9.80665 if converted else height_m)
    path = _write_orography(tmp_path / "orog.nc", LATS, LONS, values, name=name)
    if units is not None:
        with xr.open_dataset(path) as ds:
            ds = ds.load()
        ds[name].attrs["units"] = units
        ds.to_netcdf(tmp_path / "orog_units.nc")
        path = str(tmp_path / "orog_units.nc")
    _, _, orog = load_orography_grid(path)
    np.testing.assert_allclose(orog, height_m)


def test_load_orography_grid_unknown_variable(tmp_path):
    path = _write_orography(tmp_path / "t2m.nc", LATS, LONS, np.zeros((len(LATS), len(LONS))), name="t2m")
    with pytest.raises(ValueError, match="Cannot find orography variable"):
        load_orography_grid(path)


def test_elevation_at_peak_and_outside(orography_file):
    elevation = compute_orography_elevation(orography_file, _rad([[46.0, 8.0], [46.4, 8.0], [0.0, 0.0]]))
    assert elevation[0] == pytest.approx(3000.0, rel=1e-6)
    assert 0 < elevation[1] < elevation[0]
    assert elevation[2] == 0.0  # outside the orography grid


def test_elevation_longitude_conventions(tmp_path):
    """Grids in [0, 360) are queried correctly with nodes in [-180, 180)."""
    lons = np.arange(350.0, 359.0001, 0.5)
    lat_g, lon_g = np.meshgrid(LATS, lons, indexing="ij")
    path = _write_orography(tmp_path / "east.nc", LATS, lons, lon_g)
    elevation = compute_orography_elevation(path, _rad([[46.0, -5.0]]))
    assert elevation[0] == pytest.approx(355.0)


def test_gradient_of_linear_ramp(tmp_path):
    """A ramp rising 1 m per 100 m northward has a slope of 0.01 everywhere inside the grid."""
    lat_g, _ = np.meshgrid(LATS, LONS, indexing="ij")
    ramp = 0.01 * np.deg2rad(lat_g - 44.0) * EARTH_RADIUS_M
    path = _write_orography(tmp_path / "ramp.nc", LATS, LONS, ramp)
    gradient = compute_orography_gradient(path, _rad([[45.0, 7.0], [46.0, 8.0], [47.0, 9.0]]))
    np.testing.assert_allclose(gradient, 0.01, rtol=1e-6)


def test_gradient_flat_and_mountain(orography_file, flat_orography_file):
    coords = _rad([[46.0, 8.0], [46.4, 8.0], [47.9, 9.9]])
    np.testing.assert_allclose(compute_orography_gradient(flat_orography_file, coords), 0.0, atol=1e-12)

    gradient = compute_orography_gradient(orography_file, coords)
    # Zero slope at the summit, steep on the flank, nearly flat far away.
    assert gradient[0] < 1e-3
    assert gradient[1] > 10 * gradient[2]
