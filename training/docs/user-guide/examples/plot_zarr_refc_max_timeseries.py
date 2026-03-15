#!/usr/bin/env python3
"""Plot max reflectivity (refc) time series from an Anemoi Zarr dataset.

This script scans the selected variable over all non-time dimensions
(node/grid/ensemble/etc.) and computes one max value per valid time.
Optionally, it can also compute a high-percentile (default: p95) curve.

Example:
  python training/docs/user-guide/examples/plot_zarr_refc_max_timeseries.py \
    /path/to/rrfs-conus-3km-202405-bcmask-time-s.zarr \
    --variable refc \
    --start 2024-05-02T09:00:00 \
    --end 2024-05-31T20:00:00 \
    --out-png refc_max_timeseries.png \
    --out-csv refc_max_timeseries.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _pick_time_coord(ds: xr.Dataset) -> str:
    for name in ("time", "dates"):
        if name in ds.coords:
            return name
    for name in ("time", "dates"):
        if name in ds.variables and ds[name].ndim == 1:
            return name
    raise ValueError("Could not find time coordinate. Expected 'time' or 'dates'.")


def _pick_data_var(ds: xr.Dataset) -> str:
    if "data" in ds.data_vars:
        return "data"
    # Fallback to first non-scalar data variable.
    candidates = [k for k, v in ds.data_vars.items() if v.ndim >= 2]
    if not candidates:
        raise ValueError("Could not find a suitable data variable (expected 'data').")
    return candidates[0]


def _pick_variable_dim(arr: xr.DataArray) -> str:
    for d in ("variable", "vars", "channel", "feature"):
        if d in arr.dims:
            return d
    raise ValueError(f"No variable dimension found in data dims: {arr.dims}")


def _is_numeric_strings(values: list[str]) -> bool:
    return all(v.isdigit() for v in values)


def _variable_names(ds: xr.Dataset, data_var: str, variable_dim: str) -> list[str]:
    names: list[str] | None = None
    if variable_dim in ds.coords:
        names = [str(v) for v in ds.coords[variable_dim].values]
    elif "variable" in ds.coords:
        names = [str(v) for v in ds.coords["variable"].values]

    # Some zarrs store numeric indices in the coord, with real names in attrs.
    if names and not _is_numeric_strings(names):
        return names

    attr_candidates = [
        ds[data_var].attrs.get("variables"),
        ds.attrs.get("variables"),
    ]
    for candidate in attr_candidates:
        if isinstance(candidate, (list, tuple)):
            c = [str(v) for v in candidate]
            if len(c) == int(ds[data_var].sizes[variable_dim]):
                return c

    meta = ds.attrs.get("variables_metadata")
    if isinstance(meta, dict):
        c = [str(k) for k in meta.keys()]
        if len(c) == int(ds[data_var].sizes[variable_dim]):
            return c

    if names:
        return names
    raise ValueError("Variable names not found in coords or attrs.")


def _variable_index(names: list[str], variable_name: str) -> int:

    if variable_name not in names:
        sample = names[:20]
        tail = "..." if len(names) > 20 else ""
        raise ValueError(f"Variable '{variable_name}' not found. Available: {sample}{tail}")
    return names.index(variable_name)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot per-valid-time max of a variable from Anemoi Zarr.")
    p.add_argument("dataset", type=Path, help="Path to Zarr dataset.")
    p.add_argument("--variable", default="refc", help="Variable name to analyze (default: refc).")
    p.add_argument("--variable-index", type=int, default=None, help="Use explicit variable index instead of --variable.")
    p.add_argument("--list-variables", action="store_true", help="Print resolved variable names and exit.")
    p.add_argument("--start", default=None, help="Optional start time (inclusive), e.g. 2024-05-02T09:00:00.")
    p.add_argument("--end", default=None, help="Optional end time (inclusive), e.g. 2024-05-31T20:00:00.")
    p.add_argument("--out-png", type=Path, default=Path("refc_max_timeseries.png"), help="Output PNG path.")
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("refc_max_timeseries.csv"),
        help="Output CSV path with columns: valid_time,max_value[,pXX_value]",
    )
    p.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Optional percentile curve to compute (0-100). Default: 95.",
    )
    p.add_argument("--title", default=None, help="Optional custom plot title.")
    args = p.parse_args()

    ds = xr.open_zarr(args.dataset, consolidated=False)
    time_coord = _pick_time_coord(ds)
    data_var = _pick_data_var(ds)
    arr = ds[data_var]

    variable_dim = _pick_variable_dim(arr)
    names = _variable_names(ds, data_var, variable_dim)
    if args.list_variables:
        print("\n".join(f"{i:4d} {name}" for i, name in enumerate(names)))
        return

    if args.variable_index is not None:
        vidx = args.variable_index
        if vidx < 0 or vidx >= len(names):
            raise ValueError(f"--variable-index {vidx} out of range [0, {len(names)-1}]")
    else:
        vidx = _variable_index(names, args.variable)
    arr = arr.isel({variable_dim: vidx})

    # Apply optional time range.
    if args.start is not None or args.end is not None:
        start = args.start if args.start is not None else None
        end = args.end if args.end is not None else None
        arr = arr.sel({time_coord: slice(start, end)})

    if arr.sizes.get(time_coord, 0) == 0:
        raise ValueError("No time samples remain after selection.")

    reduce_dims = [d for d in arr.dims if d != time_coord]
    series = arr.max(dim=reduce_dims, skipna=True).compute()
    q = args.percentile
    if q is not None:
        if q < 0.0 or q > 100.0:
            raise ValueError(f"--percentile must be in [0,100], got {q}")
        series_q = arr.quantile(q / 100.0, dim=reduce_dims, skipna=True).compute()
        vals_q = np.asarray(series_q.values, dtype=np.float64)
    else:
        vals_q = None

    times = np.asarray(series[time_coord].values)
    vals = np.asarray(series.values, dtype=np.float64)

    # Save CSV.
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8") as f:
        if vals_q is None:
            f.write("valid_time,max_value\n")
        else:
            pcol = f"p{int(round(q)):02d}_value"
            f.write(f"valid_time,max_value,{pcol}\n")
        for i, (t, v) in enumerate(zip(times, vals, strict=True)):
            t_str = np.datetime_as_string(np.datetime64(t), unit="s")
            if vals_q is None:
                f.write(f"{t_str},{v:.10g}\n")
            else:
                f.write(f"{t_str},{v:.10g},{vals_q[i]:.10g}\n")

    # Plot.
    fig, ax = plt.subplots(figsize=(12, 4.5), layout="tight")
    ax.plot(times, vals, linewidth=1.2, label="max")
    if vals_q is not None:
        ax.plot(times, vals_q, linewidth=1.2, label=f"p{int(round(q)):02d}")
    ax.set_xlabel("valid time")
    ax.set_ylabel(f"max({args.variable})")
    ax.grid(True, alpha=0.3)
    ax.set_title(args.title or f"Max {args.variable} vs valid time")
    if vals_q is not None:
        ax.legend()
    fig.autofmt_xdate()

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=150)
    plt.close(fig)

    peak_idx = int(np.nanargmax(vals))
    peak_time = np.datetime_as_string(np.datetime64(times[peak_idx]), unit="s")
    peak_val = float(vals[peak_idx])

    print(f"Dataset      : {args.dataset}")
    print(f"Data variable: {data_var}")
    print(f"Time coord   : {time_coord}")
    print(f"Variable     : {args.variable}")
    print(f"Var index    : {vidx}")
    print(f"Samples      : {vals.size}")
    print(f"Peak         : {peak_val:.6g} at {peak_time}")
    if vals_q is not None:
        q_idx = int(np.nanargmax(vals_q))
        q_time = np.datetime_as_string(np.datetime64(times[q_idx]), unit="s")
        q_val = float(vals_q[q_idx])
        print(f"P{int(round(q)):02d} peak   : {q_val:.6g} at {q_time}")
    print(f"Wrote CSV    : {args.out_csv}")
    print(f"Wrote PNG    : {args.out_png}")


if __name__ == "__main__":
    main()
