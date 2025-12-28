#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import zarr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add/replace datetime coordinates in an Anemoi Zarr dataset."
    )
    parser.add_argument("input", type=Path, help="Input Zarr path")
    parser.add_argument("output", type=Path, help="Output Zarr path")
    parser.add_argument("--start", required=True, help="Start datetime, e.g. 2024-05-05T00:00:00")
    parser.add_argument("--frequency", required=True, help="Frequency, e.g. 1h")
    return parser.parse_args()


def main():
    args = parse_args()
    ds = xr.open_zarr(args.input, consolidated=False)
    if "data" not in ds:
        raise SystemExit("Expected 'data' variable in the Zarr dataset.")

    time_dim = "time" if "time" in ds["data"].dims else ds["data"].dims[0]
    n_time = ds["data"].sizes[time_dim]

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

    ds_out = ds.copy()
    ds_out = ds_out.assign_coords({time_dim: dates})
    ds_out["dates"] = xr.DataArray(dates, dims=(time_dim,))

    ds_out.attrs["start_date"] = str(dates[0])
    ds_out.attrs["end_date"] = str(dates[-1])
    ds_out.attrs["frequency"] = args.frequency

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_zarr(args.output, mode="w")

    # Force-update the dates array in the zarr store to datetime64
    g = zarr.open_group(args.output, mode="a")
    if "dates" in g:
        g["dates"][:] = dates.astype("datetime64[ns]")
    else:
        g.create_dataset("dates", data=dates.astype("datetime64[ns]"), overwrite=True)

    g.attrs["start_date"] = str(dates[0])
    g.attrs["end_date"] = str(dates[-1])
    g.attrs["frequency"] = args.frequency

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
