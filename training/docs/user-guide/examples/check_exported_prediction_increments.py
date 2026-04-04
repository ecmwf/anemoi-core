#!/usr/bin/env python3
"""
Numerically inspect exported Anemoi prediction files.

This compares:
  - prediction - last input
  - target - last input
  - target - prediction

so it is easy to tell whether the model is behaving like persistence.

Example:
  python training/docs/user-guide/examples/check_exported_prediction_increments.py \
    /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/verify/predictions/pred_target_epoch000_batch0000.nc \
    --variable temp_850
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def _open_export(path: Path) -> xr.Dataset:
    if path.suffix == ".nc":
        return xr.open_dataset(path)
    if path.suffix == ".zarr":
        return xr.open_zarr(path, consolidated=False)
    raise ValueError(f"Unsupported export file: {path}")


def _select_var(ds: xr.Dataset, var: str) -> int:
    if "variable" not in ds.coords:
        raise ValueError("Export file is missing 'variable' coordinate.")
    names = list(ds.coords["variable"].values)
    if var not in names:
        raise ValueError(f"Variable '{var}' not found. Available: {names}")
    return names.index(var)


def _stats(name: str, arr: np.ndarray) -> str:
    finite = np.isfinite(arr)
    if not finite.any():
        return f"{name}: all-nonfinite"
    vals = arr[finite]
    rms = float(np.sqrt(np.mean(vals**2)))
    mean_abs = float(np.mean(np.abs(vals)))
    return (
        f"{name}: mean={float(np.mean(vals)):.6g} "
        f"mean_abs={mean_abs:.6g} rms={rms:.6g} "
        f"min={float(np.min(vals)):.6g} max={float(np.max(vals)):.6g}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("export", type=Path, help="Exported NetCDF/Zarr from ExportPredictions.")
    p.add_argument("--variable", required=True, help="Variable name to inspect.")
    p.add_argument(
        "--time-index",
        type=int,
        default=None,
        help="Optional single target/pred time index. Default: inspect all available times.",
    )
    args = p.parse_args()

    ds = _open_export(args.export)
    var_idx = _select_var(ds, args.variable)

    inp = ds["input"].values
    targ = ds["target"].values
    pred = ds["prediction"].values

    # ExportPredictions writes:
    # input: (input_time, node, variable)
    # target/prediction: (target_time|pred_time, node, variable)
    inp = inp[:, :, var_idx]
    targ = targ[:, :, var_idx]
    pred = pred[:, :, var_idx]

    last_input = inp[-1]

    if args.time_index is None:
        indices = range(min(targ.shape[0], pred.shape[0]))
    else:
        indices = [args.time_index]

    print(f"file: {args.export}")
    print(f"variable: {args.variable}")
    print(f"input_time[-1]: {str(ds['input_time'].values[-1])}")

    for i in indices:
        target_time = str(ds["target_time"].values[i])
        pred_time = str(ds["pred_time"].values[i])
        targ_i = targ[i]
        pred_i = pred[i]

        pred_minus_input = pred_i - last_input
        target_minus_input = targ_i - last_input
        target_minus_pred = targ_i - pred_i

        print()
        print(f"time_index={i} target_time={target_time} pred_time={pred_time}")
        print(_stats("pred-input", pred_minus_input))
        print(_stats("target-input", target_minus_input))
        print(_stats("target-pred", target_minus_pred))

        denom = np.mean(np.abs(target_minus_input[np.isfinite(target_minus_input)]))
        numer = np.mean(np.abs(pred_minus_input[np.isfinite(pred_minus_input)]))
        if np.isfinite(denom) and denom > 0:
            print(f"ratio mean_abs(pred-input) / mean_abs(target-input) = {numer / denom:.6g}")
        else:
            print("ratio mean_abs(pred-input) / mean_abs(target-input) = n/a")


if __name__ == "__main__":
    main()
