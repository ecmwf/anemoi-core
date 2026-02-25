#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Add a boundary_mask variable to an Anemoi Zarr dataset, and optionally "
            "rewrite time/dates coordinates in one pass."
        )
    )
    parser.add_argument("input", type=Path, help="Input Zarr path")
    parser.add_argument("output", type=Path, help="Output Zarr path")
    # Default to current RRFS subdomain bounds used in this repo workflow.
    parser.add_argument("--lon-min", type=float, default=-105.0)
    parser.add_argument("--lon-max", type=float, default=-90.0)
    parser.add_argument("--lat-min", type=float, default=25.0)
    parser.add_argument("--lat-max", type=float, default=40.0)
    parser.add_argument("--boundary-km", type=float, default=20.0)
    parser.add_argument("--var-name", type=str, default="boundary_mask")
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional start datetime (e.g. 2024-05-05T00:00:00) to rewrite time/dates.",
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default=None,
        help="Optional frequency (e.g. 1h or 60m) used with --start to rewrite time/dates.",
    )
    parser.add_argument(
        "--time-unit",
        default="s",
        choices=["s", "ns"],
        help="Datetime storage unit when rewriting time/dates (default: s).",
    )
    return parser.parse_args()


def _parse_frequency_to_timedelta64(freq: str) -> np.timedelta64:
    freq = freq.strip().lower()
    if freq.endswith("h"):
        return np.timedelta64(int(freq[:-1]), "h")
    if freq.endswith("m"):
        return np.timedelta64(int(freq[:-1]), "m")
    raise ValueError(f"Unsupported frequency '{freq}'. Expected suffix h or m.")


def main():
    args = parse_args()

    ds = xr.open_zarr(args.input, consolidated=False)
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

    boundary = np.where(min_dist <= args.boundary_km, 1.0, 0.0).astype(np.float32)

    data = ds["data"]
    if "variable" not in data.dims:
        raise SystemExit("Expected 'variable' dimension on data.")

    time_dim = "time"
    ens_dim = "ensemble"
    if time_dim not in data.dims or ens_dim not in data.dims or "variable" not in data.dims:
        raise SystemExit("Expected data dims to include time, variable, ensemble.")

    node_candidates = [d for d in data.dims if d not in {time_dim, ens_dim, "variable"}]
    if len(node_candidates) != 1:
        raise SystemExit(f"Could not infer node dimension from data dims: {data.dims}")
    node_dim = node_candidates[0]

    boundary_flat = boundary.reshape(-1)
    if boundary_flat.size != data.sizes[node_dim]:
        raise SystemExit(
            f"Boundary mask size {boundary_flat.size} does not match data[{node_dim}] "
            f"size {data.sizes[node_dim]}."
        )

    mask_da = xr.DataArray(
        np.broadcast_to(
            boundary_flat.reshape(1, 1, -1),
            (data.sizes[time_dim], data.sizes[ens_dim], data.sizes[node_dim]),
        ),
        dims=(time_dim, ens_dim, node_dim),
        coords={
            time_dim: data[time_dim],
            ens_dim: data[ens_dim],
            node_dim: data[node_dim],
        },
    ).expand_dims(variable=[args.var_name])

    mask_da = mask_da.transpose(*data.dims)

    var_coord = None
    if "variable" in data.coords:
        var_coord = list(data.coords["variable"].values)

    if var_coord is not None and args.var_name in var_coord:
        raise SystemExit(f"Variable '{args.var_name}' already exists.")

    core_coords = {
        time_dim: data[time_dim],
        ens_dim: data[ens_dim],
        node_dim: data[node_dim],
    }
    data_core = xr.DataArray(data.data, dims=data.dims, coords=core_coords)
    mask_core = xr.DataArray(mask_da.data, dims=data.dims, coords=core_coords)

    new_data = xr.concat([data_core, mask_core], dim="variable")
    if var_coord is not None:
        new_data = new_data.assign_coords(variable=var_coord + [args.var_name])
    else:
        new_data = new_data.assign_coords(variable=np.arange(new_data.sizes["variable"]))

    # Extend any per-variable stats arrays to avoid alignment errors.
    var_dim_vars = [v for v in ds.data_vars if "variable" in ds[v].dims]
    ds_out = ds.drop_vars(var_dim_vars)

    for v in var_dim_vars:
        if v == "data":
            ds_out["data"] = new_data
            continue

        arr = ds[v]
        if arr.dims.count("variable") != 1:
            ds_out[v] = arr
            continue

        fill = 0.0
        if v == "stdev":
            fill = 1.0
        if v == "has_nans":
            fill = 1
        if v == "count":
            fill = 0

        arr_np = np.array(arr)
        var_axis = arr.dims.index("variable")
        pad_shape = list(arr_np.shape)
        pad_shape[var_axis] = 1
        pad = np.full(pad_shape, fill, dtype=arr_np.dtype)
        arr_new_np = np.concatenate([arr_np, pad], axis=var_axis)

        coords = {dim: arr.coords[dim] for dim in arr.dims if dim in arr.coords}
        if var_coord is not None:
            coords["variable"] = var_coord + [args.var_name]
        arr_new = xr.DataArray(arr_new_np, dims=arr.dims, coords=coords)
        ds_out[v] = arr_new

    for obj in (ds_out, ds_out["data"]):
        vars_attr = obj.attrs.get("variables")
        if isinstance(vars_attr, (list, tuple)):
            obj.attrs["variables"] = list(vars_attr) + [args.var_name]

    vars_meta = ds_out.attrs.get("variables_metadata")
    if isinstance(vars_meta, dict) and args.var_name not in vars_meta:
        vars_meta[args.var_name] = {"constant_in_time": True, "mars": {}}
        ds_out.attrs["variables_metadata"] = vars_meta

    if (args.start is None) != (args.frequency is None):
        raise SystemExit("Use --start and --frequency together, or neither.")
    if args.start is not None and args.frequency is not None:
        data_out = ds_out["data"]
        if "time" not in data_out.dims:
            raise SystemExit("Expected 'time' dimension in data.")
        n_time = data_out.sizes["time"]

        start = np.datetime64(args.start)
        step = _parse_frequency_to_timedelta64(args.frequency)
        dates = start + np.arange(n_time) * step
        dt64 = dates.astype(f"datetime64[{args.time_unit}]")

        ds_out["dates"] = xr.DataArray(dt64, dims=("time",))
        ds_out["time"] = xr.DataArray(dt64, dims=("time",))

        ds_out.attrs["start_date"] = str(dt64[0])
        ds_out.attrs["end_date"] = str(dt64[-1])
        ds_out.attrs["frequency"] = args.frequency

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_zarr(args.output, mode="w")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
