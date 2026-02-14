#!/usr/bin/env python3
"""
Plot exported predictions/targets from ExportPredictions NetCDF/Zarr.

Examples:
  python plot_exported_predictions.py /path/to/pred_target_epoch000_batch0000.nc \
    --dataset /path/to/test-20km-bcmask-time.zarr \
    --variable temp_850 --time-index 0 --out temp_850_t0.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr

# Headless backend for HPC / no X11.
matplotlib.use("Agg")


def _open_export(path: Path) -> xr.Dataset:
    if path.suffix == ".nc":
        return xr.open_dataset(path)
    if path.suffix == ".zarr":
        return xr.open_zarr(path, consolidated=False)
    raise ValueError(f"Unsupported export file: {path}")


def _export_latlon(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray] | None:
    if "latitude" in ds.coords and "longitude" in ds.coords:
        return np.asarray(ds.coords["latitude"].values).reshape(-1), np.asarray(ds.coords["longitude"].values).reshape(-1)
    return None


def _open_latlon(dataset_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        ds = xr.open_zarr(dataset_path, consolidated=False)
    except Exception:
        return None

    lat = None
    lon = None
    if "latitudes" in ds:
        lat = ds["latitudes"].values
    elif "latitude" in ds.coords:
        lat = ds.coords["latitude"].values

    if "longitudes" in ds:
        lon = ds["longitudes"].values
    elif "longitude" in ds.coords:
        lon = ds.coords["longitude"].values

    if lat is None or lon is None:
        return None
    return np.asarray(lat).reshape(-1), np.asarray(lon).reshape(-1)


def _align_node_length(
    targ: np.ndarray,
    pred: np.ndarray,
    err: np.ndarray,
    latlon: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
    if latlon is None:
        return targ, pred, err, latlon
    lat, lon = latlon
    n_data = min(targ.shape[0], pred.shape[0], err.shape[0])
    n_ll = min(lat.shape[0], lon.shape[0])
    if n_data != n_ll:
        raise ValueError(
            f"Node-count mismatch: data has {n_data} nodes but lat/lon has {n_ll}. "
            "Use export-file latitude/longitude (preferred) or provide matching dataset."
        )
    return targ[:n_data], pred[:n_data], err[:n_data], (lat[:n_data], lon[:n_data])


def _select_var(ds: xr.Dataset, var: str) -> int:
    if "variable" not in ds.coords:
        raise ValueError("Export file is missing 'variable' coordinate.")
    names = list(ds.coords["variable"].values)
    if var not in names:
        raise ValueError(f"Variable '{var}' not found. Available: {names[:20]}{'...' if len(names) > 20 else ''}")
    return names.index(var)


def _get_time(ds: xr.Dataset, time_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    targ = ds["target"].isel(target_time=time_index)
    pred = ds["prediction"].isel(pred_time=time_index)
    err = targ - pred
    time_val = str(ds["target_time"].values[time_index])
    return targ.values, pred.values, err.values, time_val


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("export", type=Path, help="Exported NetCDF/Zarr from ExportPredictions.")
    p.add_argument("--dataset", type=Path, default=None, help="Zarr dataset for lat/lon lookup (optional).")
    p.add_argument(
        "--prefer-dataset-latlon",
        action="store_true",
        help="Use dataset lat/lon even if export file has latitude/longitude.",
    )
    p.add_argument("--variable", required=True, help="Variable name (must match export 'variable' coord).")
    p.add_argument("--time-index", type=int, default=0, help="Target/pred time index to plot.")
    p.add_argument("--out", type=Path, default=Path("pred_target_compare.png"), help="Output PNG path.")
    args = p.parse_args()

    ds = _open_export(args.export)
    var_idx = _select_var(ds, args.variable)
    targ, pred, err, time_val = _get_time(ds, args.time_index)

    # Select variable along last axis (node, variable)
    targ = targ[:, var_idx]
    pred = pred[:, var_idx]
    err = err[:, var_idx]

    latlon = None
    latlon_source = "none"
    if args.prefer_dataset_latlon and args.dataset:
        latlon = _open_latlon(args.dataset)
        latlon_source = "dataset"
    else:
        latlon = _export_latlon(ds)
        if latlon is not None:
            latlon_source = "export"
        elif args.dataset:
            latlon = _open_latlon(args.dataset)
            latlon_source = "dataset"

    if latlon is not None:
        print(f"Using lat/lon from: {latlon_source} (n={latlon[0].shape[0]})")
    else:
        print("No lat/lon available; plotting by node index.")
    targ, pred, err, latlon = _align_node_length(targ, pred, err, latlon)

    import matplotlib.pyplot as plt  # local import to avoid backend issues

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), layout="tight")
    titles = [
        f"target {args.variable}\n{time_val}",
        f"prediction {args.variable}\n{time_val}",
        f"error (target-pred)\n{time_val}",
    ]

    if latlon is None:
        # Fallback: line plot over node index
        x = np.arange(targ.shape[0])
        axs[0].plot(x, targ, linewidth=0.5)
        axs[1].plot(x, pred, linewidth=0.5)
        axs[2].plot(x, err, linewidth=0.5)
        for ax, title in zip(axs, titles):
            ax.set_title(title)
            ax.set_xlabel("node")
    else:
        lat, lon = latlon
        sc0 = axs[0].scatter(lon, lat, c=targ, s=1)
        sc1 = axs[1].scatter(lon, lat, c=pred, s=1)
        sc2 = axs[2].scatter(lon, lat, c=err, s=1, cmap="bwr")
        for ax, title in zip(axs, titles):
            ax.set_title(title)
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
        fig.colorbar(sc0, ax=axs[0], shrink=0.8)
        fig.colorbar(sc1, ax=axs[1], shrink=0.8)
        fig.colorbar(sc2, ax=axs[2], shrink=0.8)

    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
