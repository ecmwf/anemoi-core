# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest

from anemoi.training.diagnostics.evaluation.zonal import EARTH_RADIUS
from anemoi.training.diagnostics.evaluation.zonal import GRAVITY
from anemoi.training.diagnostics.evaluation.zonal import OMEGA
from anemoi.training.diagnostics.evaluation.zonal import R_DRY
from anemoi.training.diagnostics.evaluation.zonal import ZonalGrid
from anemoi.training.diagnostics.evaluation.zonal import hadley
from anemoi.training.diagnostics.evaluation.zonal import jet
from anemoi.training.diagnostics.evaluation.zonal import peak
from anemoi.training.diagnostics.evaluation.zonal import section_rmse
from anemoi.training.diagnostics.evaluation.zonal import streamfunction
from anemoi.training.diagnostics.evaluation.zonal import thermal_wind
from anemoi.training.diagnostics.evaluation.zonal import zonal_moments

ROWS = np.linspace(-89.0, 89.0, 90)


def _reduced_grid() -> tuple[np.ndarray, np.ndarray]:
    lats, lons = [], []
    for lat in ROWS:
        n_lon = int(16 + 40 * np.cos(np.deg2rad(lat)))
        lats.extend([lat] * n_lon)
        lons.extend(np.linspace(0.0, 360.0, n_lon, endpoint=False))
    return np.asarray(lats), np.deg2rad(np.asarray(lons))


LAT, LAM = _reduced_grid()


def test_zonal_mean_removes_waves_and_honours_mask() -> None:
    grid = ZonalGrid(LAT)
    np.testing.assert_allclose(grid.lat, ROWS)
    field = np.sin(np.deg2rad(LAT)) + 5.0 * np.cos(3 * LAM)
    np.testing.assert_allclose(grid.mean(field)[0], np.sin(np.deg2rad(ROWS)), atol=1e-12)

    mask = (LAT > 0) | (np.pi > LAM)  # SH rows keep only the first half of the longitudes
    masked = grid.mean(np.where(np.pi > LAM, 1.0, 9.0), mask)[0]
    np.testing.assert_allclose(masked[ROWS < 0], 1.0)
    assert np.all(grid.fraction(mask)[0][ROWS < 0] == pytest.approx(0.5, abs=0.03))
    assert np.all(np.isnan(grid.mean(field, LAT > 0)[0][ROWS < 0]))


def test_eddy_fluxes_of_correlated_waves() -> None:
    grid = ZonalGrid(LAT)
    wave = np.cos(2 * LAM)
    u, v, t = (10.0 + 4.0 * wave)[None], (3.0 * wave)[None], (280.0 - 2.0 * wave)[None]
    moments = zonal_moments(grid, u, v, t, None)
    np.testing.assert_allclose(moments["uv"][0], 4.0 * 3.0 / 2, atol=1e-10)
    np.testing.assert_allclose(moments["vt"][0], -3.0 * 2.0 / 2, atol=1e-10)
    np.testing.assert_allclose(moments["frac"], 1.0)


def test_thermal_wind_balance_of_an_exact_solution() -> None:
    lat = np.linspace(-89.0, 89.0, 357)
    levels = np.array([250.0, 500.0, 850.0])
    delta_t = 30.0
    t = 280.0 - delta_t * np.sin(np.deg2rad(lat)) ** 2 - np.array([60.0, 30.0, 0.0])[:, None]
    shear = R_DRY * delta_t * np.cos(np.deg2rad(lat)) * np.log(850.0 / 250.0) / (OMEGA * EARTH_RADIUS)
    u = np.stack([shear, 0.5 * shear, np.zeros_like(lat)])

    tw = thermal_wind(lat, levels, t, u, (850, 250))
    off = np.abs(lat) >= 15.0
    assert np.all(np.isnan(tw["balanced"][~off]))
    inner = off & (np.abs(lat) < 89.0)  # one-sided gradient at the end rows
    np.testing.assert_allclose(tw["balanced"][inner], tw["actual"][inner], rtol=1e-3, atol=1e-3)
    assert 15.0 < tw["balanced"][np.argmin(np.abs(lat - 30.0))] < 30.0  # ~ observed subtropical shear


def test_streamfunction_recovers_prescribed_cells() -> None:
    lat = np.linspace(-80.0, 80.0, 81)
    levels = np.linspace(50.0, 1000.0, 200)
    p_pa = 100.0 * levels
    shape = np.sin(np.pi * (levels - 50.0) / 950.0)[:, None]
    psi = 1e11 * shape * np.sin(np.deg2rad(3 * lat))[None]
    dpsi_dp = np.gradient(psi, p_pa, axis=0)
    v = GRAVITY / (2 * np.pi * EARTH_RADIUS * np.cos(np.deg2rad(lat))) * dpsi_dp

    out = streamfunction(lat, levels, v)
    assert np.max(np.abs(out - psi)) < 1e-3 * 1e11
    np.testing.assert_allclose(out[[0, -1]], 0.0, atol=1e-6 * 1e11)
    north = np.argmin(np.abs(lat - 15.0))
    assert out[100, north] > 0 > out[100, -north - 1]

    # A uniform mass inflow is removed with the column mean; zero-fraction levels carry no mass.
    frac = np.ones_like(v)
    frac[-50:] = 0.0
    shifted = streamfunction(lat, levels, np.where(frac > 0, v + 2.0, np.nan), frac)
    np.testing.assert_allclose(shifted[-1], 0.0, atol=1e-6 * 1e11)


def test_jet_hadley_and_peak_extractors() -> None:
    lat = np.arange(-89.0, 90.0, 2.0)
    u = 30.0 * np.exp(-((lat - 37.3) ** 2) / 60.0) + 45.0 * np.exp(-((lat + 50.6) ** 2) / 60.0)
    jets = jet(lat, u)
    assert jets["nh"][0] == pytest.approx(37.3, abs=0.2)
    assert jets["nh"][1] == pytest.approx(30.0, abs=0.3)
    assert jets["sh"][0] == pytest.approx(-50.6, abs=0.2)

    psi = 1e11 * np.sin(np.deg2rad(6 * lat)) * (np.abs(lat) < 60)
    cells = hadley(lat, psi)
    assert cells["nh"][0] > 0.9e11
    assert cells["sh"][0] < -0.9e11
    assert cells["nh"][1] == pytest.approx(30.0, abs=0.1)
    assert cells["sh"][1] == pytest.approx(-30.0, abs=0.1)
    assert np.isnan(hadley(lat, np.abs(psi) + 1.0)["nh"][1])

    flux = 10.0 * np.sin(np.deg2rad(2 * lat))
    peaks = peak(lat, flux)
    assert peaks["nh"] == pytest.approx(10.0, abs=0.01)
    assert peaks["sh"] == pytest.approx(-10.0, abs=0.01)


def test_section_rmse_weights_and_ignores_nan() -> None:
    lat = np.array([-60.0, 0.0, 60.0])
    diff = np.array([[1.0, 2.0, np.nan]])
    weights = np.cos(np.deg2rad(lat[:2]))
    assert section_rmse(lat, diff) == pytest.approx(np.sqrt(np.sum(weights * [1.0, 4.0]) / weights.sum()))
