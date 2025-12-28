#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add a boundary_mask variable to an Anemoi Zarr dataset."
    )
    parser.add_argument("input", type=Path, help="Input Zarr path")
    parser.add_argument("output", type=Path, help="Output Zarr path")
    parser.add_argument("--lon-min", type=float, required=True)
    parser.add_argument("--lon-max", type=float, required=True)
    parser.add_argument("--lat-min", type=float, required=True)
    parser.add_argument("--lat-max", type=float, required=True)
    parser.add_argument("--boundary-km", type=float, default=20.0)
    parser.add_argument("--var-name", type=str, default="boundary_mask")
    return parser.parse_args()


def main():
    args = parse_args()

    ds = xr.open_zarr(args.input)
    if "data" not in ds:
        raise SystemExit("Expected 'data' variable in the Zarr dataset.")

    lat = ds["latitudes"].values
    lon = ds["longitudes"].values
    if lat.shape != lon.shape:
        raise SystemExit("latitudes/longitudes shapes do not match.")

    lat_min = args.lat_min
    lat_max = args.lat_max
    lon_min = args.lon_min
    lon_max = args.lon_max

    # Distance (km) to each domain edge.
    km_per_deg = 111.0
    cos_lat = np.cos(np.deg2rad(lat))
    dy_south = (lat - lat_min) * km_per_deg
    dy_north = (lat_max - lat) * km_per_deg
    dx_west = (lon - lon_min) * km_per_deg * cos_lat
    dx_east = (lon_max - lon) * km_per_deg * cos_lat
    min_dist = np.minimum.reduce([dy_south, dy_north, dx_west, dx_east])

    boundary = (min_dist <= args.boundary_km).astype(np.float32)

    data = ds["data"]
    if "variable" not in data.dims:
        raise SystemExit("Expected 'variable' dimension on data.")

    time_dim = "time"
    ens_dim = "ensemble"
    node_dim = "node"

    if time_dim not in data.dims or ens_dim not in data.dims or node_dim not in data.dims:
        raise SystemExit("Expected data dims: time, variable, ensemble, node.")

    mask_da = xr.DataArray(
        np.broadcast_to(
            boundary.reshape(1, 1, -1),
            (data.sizes[time_dim], data.sizes[ens_dim], data.sizes[node_dim]),
        ),
        dims=(time_dim, ens_dim, node_dim),
        coords={
            time_dim: data[time_dim],
            ens_dim: data[ens_dim],
            node_dim: data[node_dim],
        },
    ).expand_dims(variable=[args.var_name])

    if "variable" in data.coords:
        var_coord = list(data.coords["variable"].values)
        if args.var_name in var_coord:
            raise SystemExit(f"Variable '{args.var_name}' already exists.")
        mask_da = mask_da.assign_coords(variable=var_coord + [args.var_name])

    new_data = xr.concat([data, mask_da], dim="variable")
    ds_out = ds.copy()
    ds_out["data"] = new_data

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_zarr(args.output, mode="w")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
