#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
import zarr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add/replace datetime coordinates (and optional boundary mask) in an Anemoi Zarr dataset."
    )
    parser.add_argument("input", type=Path, help="Input Zarr path")
    parser.add_argument("output", type=Path, help="Output Zarr path")
    parser.add_argument("--start", required=True, help="Start datetime, e.g. 2024-05-05T00:00:00")
    parser.add_argument("--frequency", required=True, help="Frequency, e.g. 1h")
    parser.add_argument("--lon-min", type=float)
    parser.add_argument("--lon-max", type=float)
    parser.add_argument("--lat-min", type=float)
    parser.add_argument("--lat-max", type=float)
    parser.add_argument("--boundary-km", type=float)
    parser.add_argument("--boundary-var", default="boundary_mask")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")

    # Copy the dataset first to avoid rewriting large arrays.
    shutil.copytree(args.input, args.output)

    g = zarr.open_group(args.output, mode="a")
    if "data" not in g:
        raise SystemExit("Expected 'data' array in the Zarr dataset.")

    # Anemoi datasets store data as (time, variable, ensemble, cell).
    n_time = g["data"].shape[0]

    start = np.datetime64(args.start)
    freq = np.timedelta64(1, "h")
    if args.frequency.endswith("h"):
        hours = int(args.frequency[:-1])
        freq = np.timedelta64(hours, "h")
    elif args.frequency.endswith("m"):
        mins = int(args.frequency[:-1])
        freq = np.timedelta64(mins, "m")
    else:
        raise SystemExit(f"Unsupported frequency: {args.frequency}")

    dates = start + np.arange(n_time) * freq

    # Boundary mask must already exist; avoid rewriting the main data array.
    if all(v is not None for v in [args.lon_min, args.lon_max, args.lat_min, args.lat_max, args.boundary_km]):
        vars_attr = g.attrs.get("variables", [])
        if args.boundary_var not in vars_attr:
            raise SystemExit(
                f"Boundary variable '{args.boundary_var}' not found in dataset. "
                "Run add_boundary_mask.py first."
            )

    # Force-update time/dates arrays in the zarr store to datetime64.
    dt64 = dates.astype("datetime64[ns]")
    for key in ("dates", "time"):
        if key in g:
            if g[key].dtype != dt64.dtype:
                del g[key]
                g.create_dataset(key, data=dt64, overwrite=True)
            else:
                g[key][:] = dt64
        else:
            g.create_dataset(key, data=dt64, overwrite=True)

    g.attrs["start_date"] = str(dates[0])
    g.attrs["end_date"] = str(dates[-1])
    g.attrs["frequency"] = args.frequency

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
