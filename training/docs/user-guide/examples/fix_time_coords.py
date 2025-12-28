#!/usr/bin/env python3
import argparse
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

    # Optional boundary mask
    if all(v is not None for v in [args.lon_min, args.lon_max, args.lat_min, args.lat_max, args.boundary_km]):
        lat = ds_out["latitudes"].values
        lon = ds_out["longitudes"].values
        if lat.shape != lon.shape:
            raise SystemExit("latitudes/longitudes shapes do not match.")

        km_per_deg = 111.0
        cos_lat = np.cos(np.deg2rad(lat))
        dy_south = (lat - args.lat_min) * km_per_deg
        dy_north = (args.lat_max - lat) * km_per_deg
        dx_west = (lon - args.lon_min) * km_per_deg * cos_lat
        dx_east = (args.lon_max - lon) * km_per_deg * cos_lat
        min_dist = np.minimum.reduce([dy_south, dy_north, dx_west, dx_east])

        boundary = np.where(min_dist <= args.boundary_km, 1.0, np.nan).astype(np.float32)

        data = ds_out["data"]
        node_candidates = [d for d in data.dims if d not in {time_dim, "ensemble", "variable"}]
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
                (data.sizes[time_dim], data.sizes["ensemble"], data.sizes[node_dim]),
            ),
            dims=(time_dim, "ensemble", node_dim),
            coords={time_dim: data[time_dim], "ensemble": data["ensemble"], node_dim: data[node_dim]},
        ).expand_dims(variable=[args.boundary_var])

        mask_da = mask_da.transpose(*data.dims)

        var_coord = None
        if "variable" in data.coords:
            var_coord = list(data.coords["variable"].values)
        if var_coord is not None and args.boundary_var in var_coord:
            raise SystemExit(f"Variable '{args.boundary_var}' already exists.")

        core_coords = {time_dim: data[time_dim], "ensemble": data["ensemble"], node_dim: data[node_dim]}
        data_core = xr.DataArray(data.data, dims=data.dims, coords=core_coords)
        mask_core = xr.DataArray(mask_da.data, dims=data.dims, coords=core_coords)

        new_data = xr.concat([data_core, mask_core], dim="variable")
        if var_coord is not None:
            new_data = new_data.assign_coords(variable=var_coord + [args.boundary_var])
        else:
            new_data = new_data.assign_coords(variable=np.arange(new_data.sizes["variable"]))

        var_dim_vars = [v for v in ds_out.data_vars if "variable" in ds_out[v].dims]
        ds_out = ds_out.drop_vars(var_dim_vars)

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
                coords["variable"] = var_coord + [args.boundary_var]
            arr_new = xr.DataArray(arr_new_np, dims=arr.dims, coords=coords)
            ds_out[v] = arr_new

        for obj in (ds_out, ds_out["data"]):
            vars_attr = obj.attrs.get("variables")
            if isinstance(vars_attr, (list, tuple)):
                obj.attrs["variables"] = list(vars_attr) + [args.boundary_var]

    ds_out = ds_out.assign_coords({time_dim: dates})
    ds_out["dates"] = xr.DataArray(dates, dims=(time_dim,))

    ds_out.attrs["start_date"] = str(dates[0])
    ds_out.attrs["end_date"] = str(dates[-1])
    ds_out.attrs["frequency"] = args.frequency

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_zarr(args.output, mode="w")

    # Force-update time/dates arrays in the zarr store to datetime64.
    g = zarr.open_group(args.output, mode="a")
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
