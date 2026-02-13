#!/usr/bin/env python3
"""
Plot model-node vs data-node coverage.

Inputs:
- exported prediction file produced by ExportPredictions (.nc or .zarr),
  containing node-wise latitude/longitude coordinates.
- dataset zarr containing full-grid latitudes/longitudes.

Outputs:
- PNG with three panels:
  1) full data nodes
  2) model nodes
  3) overlay (data-only in gray, model in red)

Example:
  python plot_model_vs_data_nodes.py \
    --export /path/to/pred_target_epoch000_batch0000.nc \
    --dataset /path/to/test-20km-bcmask-time-s.zarr \
    --out model_vs_data_nodes.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _open_export(path: Path) -> xr.Dataset:
    if path.suffix == ".nc":
        return xr.open_dataset(path)
    if path.suffix == ".zarr":
        return xr.open_zarr(path, consolidated=False)
    raise ValueError(f"Unsupported export format: {path}")


def _read_model_latlon(export_path: Path) -> tuple[np.ndarray, np.ndarray]:
    ds = _open_export(export_path)
    if "latitude" in ds.coords and "longitude" in ds.coords:
        lat = np.asarray(ds.coords["latitude"].values).reshape(-1)
        lon = np.asarray(ds.coords["longitude"].values).reshape(-1)
        return lat, lon
    raise ValueError(
        "Export file does not contain latitude/longitude coords. "
        "Regenerate exports with updated ExportPredictions callback."
    )


def _read_data_latlon(dataset_path: Path) -> tuple[np.ndarray, np.ndarray]:
    ds = xr.open_zarr(dataset_path, consolidated=False)
    if "latitudes" in ds and "longitudes" in ds:
        lat = np.asarray(ds["latitudes"].values).reshape(-1)
        lon = np.asarray(ds["longitudes"].values).reshape(-1)
        return lat, lon
    if "latitude" in ds.coords and "longitude" in ds.coords:
        lat = np.asarray(ds.coords["latitude"].values).reshape(-1)
        lon = np.asarray(ds.coords["longitude"].values).reshape(-1)
        return lat, lon
    raise ValueError("Could not find lat/lon in dataset zarr.")


def _quantize(lat: np.ndarray, lon: np.ndarray, decimals: int) -> np.ndarray:
    # Stable integer key per point for set-membership checks.
    scale = 10**decimals
    ilat = np.rint(lat * scale).astype(np.int64)
    ilon = np.rint(lon * scale).astype(np.int64)
    return np.stack([ilat, ilon], axis=1)


def _membership_mask(
    data_lat: np.ndarray,
    data_lon: np.ndarray,
    model_lat: np.ndarray,
    model_lon: np.ndarray,
    decimals: int,
) -> np.ndarray:
    data_key = _quantize(data_lat, data_lon, decimals)
    model_key = _quantize(model_lat, model_lon, decimals)

    # Convert to structured dtype to allow vectorized isin on 2 columns.
    dtype = np.dtype([("a", np.int64), ("b", np.int64)])
    data_struct = np.ascontiguousarray(data_key).view(dtype).reshape(-1)
    model_struct = np.ascontiguousarray(model_key).view(dtype).reshape(-1)
    return np.isin(data_struct, model_struct)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, required=True, help="Exported prediction file (.nc or .zarr).")
    ap.add_argument("--dataset", type=Path, required=True, help="Dataset zarr path.")
    ap.add_argument("--out", type=Path, default=Path("model_vs_data_nodes.png"), help="Output PNG.")
    ap.add_argument(
        "--decimals",
        type=int,
        default=6,
        help="Lat/lon matching precision in decimal places (default: 6).",
    )
    args = ap.parse_args()

    model_lat, model_lon = _read_model_latlon(args.export)
    data_lat, data_lon = _read_data_latlon(args.dataset)

    in_model = _membership_mask(
        data_lat=data_lat,
        data_lon=data_lon,
        model_lat=model_lat,
        model_lon=model_lon,
        decimals=args.decimals,
    )

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), layout="tight")

    axs[0].scatter(data_lon, data_lat, s=0.2, c="k")
    axs[0].set_title(f"Data nodes: {data_lat.size}")
    axs[0].set_xlabel("lon")
    axs[0].set_ylabel("lat")

    axs[1].scatter(model_lon, model_lat, s=0.2, c="tab:red")
    axs[1].set_title(f"Model nodes: {model_lat.size}")
    axs[1].set_xlabel("lon")
    axs[1].set_ylabel("lat")

    axs[2].scatter(data_lon[~in_model], data_lat[~in_model], s=0.2, c="0.75", label="data-only")
    axs[2].scatter(data_lon[in_model], data_lat[in_model], s=0.2, c="tab:red", label="in model")
    axs[2].set_title(
        f"Overlap: {np.count_nonzero(in_model)} / {data_lat.size}\n"
        f"Excluded: {np.count_nonzero(~in_model)}"
    )
    axs[2].set_xlabel("lon")
    axs[2].set_ylabel("lat")
    axs[2].legend(markerscale=20, loc="best")

    for ax in axs:
        ax.set_aspect("equal")

    fig.savefig(args.out, dpi=200)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

