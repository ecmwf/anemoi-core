#!/usr/bin/env python3
"""
Quick diagnostics for a lat/lon grid NetCDF file.

Usage:
  python check_latlon_grid.py rrfs-3km-subdomain-grid.nc
  python check_latlon_grid.py rrfs-3km-subdomain-grid.nc --lat-name lat --lon-name lon
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def _pick_var(ds: xr.Dataset, preferred: str, fallbacks: list[str]) -> str:
    if preferred in ds.variables:
        return preferred
    for name in fallbacks:
        if name in ds.variables:
            return name
    raise ValueError(f"Could not find variable '{preferred}' (or fallbacks: {fallbacks})")


def _summary(label: str, arr: np.ndarray) -> str:
    return (
        f"{label}: shape={arr.shape}, min={np.nanmin(arr):.6f}, "
        f"max={np.nanmax(arr):.6f}, finite={np.isfinite(arr).all()}"
    )


def _step_stats(arr: np.ndarray, axis: int) -> tuple[float, float, float]:
    d = np.diff(arr, axis=axis)
    med = float(np.nanmedian(np.abs(d)))
    p95 = float(np.nanpercentile(np.abs(d), 95))
    p99 = float(np.nanpercentile(np.abs(d), 99))
    return med, p95, p99


def _plateau_fraction(arr: np.ndarray, axis: int, tol: float = 1e-12) -> float:
    d = np.diff(arr, axis=axis)
    return float(np.mean(np.abs(d) <= tol))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("grid", type=Path, help="Grid NetCDF path")
    p.add_argument("--lat-name", default="lat", help="Latitude variable name (default: lat)")
    p.add_argument("--lon-name", default="lon", help="Longitude variable name (default: lon)")
    args = p.parse_args()

    ds = xr.open_dataset(args.grid)
    lat_name = _pick_var(ds, args.lat_name, ["latitude", "lats", "nav_lat"])
    lon_name = _pick_var(ds, args.lon_name, ["longitude", "lons", "nav_lon"])

    lat = np.asarray(ds[lat_name].values)
    lon = np.asarray(ds[lon_name].values)

    if lat.ndim != 2 or lon.ndim != 2:
        raise ValueError(f"Expected 2D lat/lon arrays. Got lat.ndim={lat.ndim}, lon.ndim={lon.ndim}")
    if lat.shape != lon.shape:
        raise ValueError(f"lat/lon shape mismatch: {lat.shape} vs {lon.shape}")

    ny, nx = lat.shape
    print(f"File: {args.grid}")
    print(_summary("lat", lat))
    print(_summary("lon", lon))
    print(f"grid size: ny={ny}, nx={nx}, n={ny * nx}")

    # Smoothness / spacing diagnostics (not strict pass/fail).
    lat_y = _step_stats(lat, axis=0)
    lat_x = _step_stats(lat, axis=1)
    lon_y = _step_stats(lon, axis=0)
    lon_x = _step_stats(lon, axis=1)
    print("abs(step) med/p95/p99:")
    print(f"  dlat(dy): {lat_y[0]:.6e} / {lat_y[1]:.6e} / {lat_y[2]:.6e}")
    print(f"  dlat(dx): {lat_x[0]:.6e} / {lat_x[1]:.6e} / {lat_x[2]:.6e}")
    print(f"  dlon(dy): {lon_y[0]:.6e} / {lon_y[1]:.6e} / {lon_y[2]:.6e}")
    print(f"  dlon(dx): {lon_x[0]:.6e} / {lon_x[1]:.6e} / {lon_x[2]:.6e}")

    # Plateau check: large plateaus are suspicious for interpolated target grids.
    p_lat_y = _plateau_fraction(lat, axis=0)
    p_lat_x = _plateau_fraction(lat, axis=1)
    p_lon_y = _plateau_fraction(lon, axis=0)
    p_lon_x = _plateau_fraction(lon, axis=1)
    print("zero-step fraction (exact plateaus):")
    print(f"  lat dy={p_lat_y:.4f}, lat dx={p_lat_x:.4f}, lon dy={p_lon_y:.4f}, lon dx={p_lon_x:.4f}")

    # Corner snapshots help detect constant corner blocks.
    k = min(5, ny, nx)
    print(f"corner sample size: {k}x{k}")
    print(f"  top-left lat std={np.nanstd(lat[:k, :k]):.6e}, lon std={np.nanstd(lon[:k, :k]):.6e}")
    print(f"  top-right lat std={np.nanstd(lat[:k, -k:]):.6e}, lon std={np.nanstd(lon[:k, -k:]):.6e}")
    print(f"  bot-left lat std={np.nanstd(lat[-k:, :k]):.6e}, lon std={np.nanstd(lon[-k:, :k]):.6e}")
    print(f"  bot-right lat std={np.nanstd(lat[-k:, -k:]):.6e}, lon std={np.nanstd(lon[-k:, -k:]):.6e}")

    warnings: list[str] = []
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        warnings.append("Found NaN/inf in lat/lon.")
    if any(v > 0.1 for v in [p_lat_y, p_lat_x, p_lon_y, p_lon_x]):
        warnings.append("Large plateau fraction detected; check for blocky/constant regions.")
    if any(np.nanstd(x) < 1e-10 for x in [lat[:k, :k], lon[:k, :k], lat[-k:, -k:], lon[-k:, -k:]]):
        warnings.append("At least one corner has nearly constant values; verify boundary construction.")

    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("No obvious structural issues found in basic checks.")


if __name__ == "__main__":
    main()

