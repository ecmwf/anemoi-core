# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Zonal-mean circulation diagnostics on pressure levels.

Cheap checks of the large-scale dynamics a forecast model should reproduce in both hemispheres:
zonal-mean wind and temperature, thermal-wind balance, the mean meridional (Hadley/Ferrel) mass
streamfunction and the eddy heat and momentum fluxes. Grids must be made of latitude rows with
equally spaced points (regular or reduced Gaussian), so a zonal mean is a plain mean over a row.

Fields are ``(level, grid)`` arrays with levels ordered from the top (lowest pressure) down.
"""

from __future__ import annotations

import numpy as np

R_DRY = 287.05  # J kg-1 K-1
EARTH_RADIUS = 6.371e6  # m
GRAVITY = 9.80665  # m s-2
OMEGA = 7.292e-5  # s-1
EQUATOR_MASK_DEG = 15.0  # thermal wind is undefined where f -> 0


class ZonalGrid:
    """Zonal means over the latitude rows of a grid.

    Parameters
    ----------
    lat : np.ndarray
        Node latitudes in degrees, shape (grid,).
    """

    def __init__(self, lat: np.ndarray) -> None:
        self.lat, self._row = np.unique(np.round(np.asarray(lat, dtype=np.float64), 6), return_inverse=True)
        self._count = np.bincount(self._row, minlength=len(self.lat)).astype(np.float64)

    def _sum(self, field: np.ndarray) -> np.ndarray:
        return np.stack([np.bincount(self._row, weights=row, minlength=len(self.lat)) for row in field])

    def mean(self, field: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Return the zonal mean of ``field`` (level, grid) over the ``mask``-ed points, (level, n_lat).

        Rows without any valid point are NaN.
        """
        field = np.atleast_2d(np.asarray(field, dtype=np.float64))
        if mask is None:
            return self._sum(field) / self._count
        mask = np.broadcast_to(mask, field.shape)
        count = self._sum(mask.astype(np.float64))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count > 0, self._sum(np.where(mask, field, 0.0)) / count, np.nan)

    def fraction(self, mask: np.ndarray) -> np.ndarray:
        """Fraction of valid points per row, (level, n_lat)."""
        return self._sum(np.atleast_2d(mask).astype(np.float64)) / self._count


def zonal_moments(grid: ZonalGrid, u: np.ndarray, v: np.ndarray, t: np.ndarray, mask: np.ndarray | None) -> dict:
    """Zonal means of u, v, t, the eddy fluxes [v'T'], [u'v'] and the valid fraction, each (level, n_lat).

    Eddies are deviations from the zonal mean of the same field, so the fluxes are
    ``[vT] - [v][T]`` (stationary plus transient eddies of one instant).
    """
    means = {name: grid.mean(field, mask) for name, field in (("u", u), ("v", v), ("t", t))}
    means["vt"] = grid.mean(v * t, mask) - means["v"] * means["t"]
    means["uv"] = grid.mean(u * v, mask) - means["u"] * means["v"]
    means["frac"] = np.ones_like(means["u"]) if mask is None else grid.fraction(mask)
    return means


def thermal_wind(lat: np.ndarray, p_hpa: np.ndarray, t: np.ndarray, u: np.ndarray, layer: tuple) -> dict:
    """Wind shear across ``layer`` (bottom, top) in hPa: actual versus thermal-wind balance.

    ``actual`` is ``[u](top) - [u](bottom)``; ``balanced`` is
    ``-(R / (f a)) dT/dphi ln(p_bottom / p_top)`` with ``T`` the ln p-weighted layer mean of [T].
    Both are NaN within ``EQUATOR_MASK_DEG`` of the equator.
    """
    p_hpa = np.asarray(p_hpa, dtype=np.float64)
    bottom, top = (int(np.nonzero(p_hpa == p)[0][0]) for p in layer)
    inside = slice(top, bottom + 1)
    log_p = np.log(p_hpa[inside])
    depth = log_p[-1] - log_p[0]
    t_layer = np.trapezoid(t[inside], log_p, axis=0) / depth
    phi = np.deg2rad(lat)
    coriolis = 2.0 * OMEGA * np.sin(phi)
    off_equator = np.abs(lat) >= EQUATOR_MASK_DEG
    with np.errstate(invalid="ignore", divide="ignore"):
        balanced = -R_DRY / (coriolis * EARTH_RADIUS) * np.gradient(t_layer, phi) * depth
    return {
        "actual": np.where(off_equator, u[top] - u[bottom], np.nan),
        "balanced": np.where(off_equator, balanced, np.nan),
    }


def streamfunction(lat: np.ndarray, p_hpa: np.ndarray, v: np.ndarray, frac: np.ndarray | None = None) -> np.ndarray:
    """Mean meridional mass streamfunction in kg s-1, (level, n_lat), zero at the top level.

    ``psi(p) = 2 pi a cos(phi) / g * int_top^p [v] dp``. Below-ground parts of a row carry no
    mass (``[v]`` is weighted by the valid ``frac``), and the column-mean [v] is removed so the
    net column mass flux, and hence psi at the bottom level, is zero.
    """
    p_pa = 100.0 * np.asarray(p_hpa, dtype=np.float64)
    frac = np.ones_like(v) if frac is None else frac
    flux = np.nan_to_num(v) * frac
    depth = np.trapezoid(frac, p_pa, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        column_mean = np.where(depth > 0, np.trapezoid(flux, p_pa, axis=0) / depth, 0.0)
    flux = flux - column_mean * frac
    integral = np.concatenate(
        (np.zeros((1, flux.shape[1])), np.cumsum(0.5 * (flux[1:] + flux[:-1]) * np.diff(p_pa)[:, None], axis=0)),
    )
    return 2.0 * np.pi * EARTH_RADIUS * np.cos(np.deg2rad(lat)) / GRAVITY * integral


def _hemispheres(lat: np.ndarray, lo: float, hi: float) -> dict[str, np.ndarray]:
    return {"nh": np.nonzero((lat >= lo) & (lat <= hi))[0], "sh": np.nonzero((lat <= -lo) & (lat >= -hi))[0]}


def jet(lat: np.ndarray, u: np.ndarray, lo: float = 20.0, hi: float = 70.0) -> dict[str, tuple[float, float]]:
    """Latitude and speed of the strongest westerly per hemisphere, refined with a parabola."""
    out = {}
    for name, index in _hemispheres(lat, lo, hi).items():
        i = int(index[np.nanargmax(u[index])])
        position, speed = lat[i], u[i]
        if 0 < i < len(lat) - 1 and np.all(np.isfinite(u[i - 1 : i + 2])):
            a, b, c = np.polyfit(lat[i - 1 : i + 2] - lat[i], u[i - 1 : i + 2], 2)
            if a < 0:
                position, speed = lat[i] - b / (2 * a), c - b**2 / (4 * a)
        out[name] = (float(position), float(speed))
    return out


def hadley(lat: np.ndarray, psi: np.ndarray, hi: float = 35.0) -> dict[str, tuple[float, float]]:
    """Hadley-cell strength (signed extremum of psi within ``hi`` deg) and poleward edge per hemisphere.

    The edge is the first zero crossing of psi poleward of the extremum, linearly interpolated;
    NaN when there is none.
    """
    out = {}
    for name, sign, step in (("nh", 1.0, 1), ("sh", -1.0, -1)):
        index = np.nonzero((sign * lat >= 0.0) & (sign * lat <= hi))[0]
        i = int(index[np.nanargmax(sign * psi[index])])
        strength = float(psi[i])
        edge = np.nan
        j = i
        while 0 <= j + step < len(lat):
            if sign * psi[j + step] <= 0.0:
                x0, x1, y0, y1 = lat[j], lat[j + step], psi[j], psi[j + step]
                edge = float(x0 + (x1 - x0) * y0 / (y0 - y1))
                break
            j += step
        out[name] = (strength, edge)
    return out


def peak(lat: np.ndarray, field: np.ndarray, lo: float = 25.0, hi: float = 75.0) -> dict[str, float]:
    """Signed value of largest magnitude within ``lo``-``hi`` deg per hemisphere."""
    return {
        name: float(field[index[np.nanargmax(np.abs(field[index]))]])
        for name, index in _hemispheres(lat, lo, hi).items()
    }


def section_rmse(lat: np.ndarray, diff: np.ndarray) -> float:
    """cos(lat)-weighted RMSE over a (level, n_lat) section, ignoring NaN."""
    weight = np.broadcast_to(np.cos(np.deg2rad(lat)), diff.shape)
    valid = np.isfinite(diff)
    return float(np.sqrt(np.sum(np.where(valid, weight * diff**2, 0.0)) / np.sum(np.where(valid, weight, 0.0))))
