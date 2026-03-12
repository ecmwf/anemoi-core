#!/usr/bin/env bash
# Run RRFS verification export and create video-ready per-valid-time frames
# for 1-hour lead prediction vs truth.
#
# Usage:
#   bash training/docs/user-guide/examples/run_ffs_verification_202505.sh \
#     <checkpoint_path> <start> <end> [frequency] [graph_path] [dataset_path]
#
# Example:
#   bash training/docs/user-guide/examples/run_ffs_verification_202505.sh \
#     /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/checkpoint/<run_id>/inference-last.ckpt \
#     2024-05-01T00:00:00 2024-05-31T23:00:00 1h
set -euo pipefail

if [[ $# -lt 3 || $# -gt 6 ]]; then
  echo "Usage: run_ffs_verification_202505.sh <checkpoint_path> <start> <end> [frequency] [graph_path] [dataset_path]"
  exit 1
fi

CHECKPOINT_PATH="$1"
START="$2"
END="$3"
FREQ="${4:-1h}"
GRAPH_PATH="${5:-}"
DATASET_PATH="${6:-/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core/tmp/rrfs-monthly/rrfs-conus-3km-202405-bcmask-time-s.zarr}"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "ERROR: checkpoint file not found: $CHECKPOINT_PATH"
  exit 2
fi

if [[ -n "$GRAPH_PATH" && ! -f "$GRAPH_PATH" ]]; then
  echo "ERROR: graph file not found: $GRAPH_PATH"
  exit 2
fi

if [[ ! -e "$DATASET_PATH" ]]; then
  echo "ERROR: dataset path not found: $DATASET_PATH"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="$SCRIPT_DIR"
CONFIG_NAME="anemoi-training-rrfs-lam-neural-lam-verify-202405"

VERIFY_ROOT="${VERIFY_ROOT:-/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/verify}"
START_TAG="$(echo "$START" | sed 's/[^0-9]//g')"
END_TAG="$(echo "$END" | sed 's/[^0-9]//g')"
RUN_TAG="ffs_verification_202505_${START_TAG}_${END_TAG}"
RUN_ROOT="${VERIFY_ROOT}/${RUN_TAG}"
EXPORT_DIR="${RUN_ROOT}/predictions"
FRAMES_ROOT="${RUN_ROOT}/video_frames"

mkdir -p "$RUN_ROOT" "$FRAMES_ROOT"

export ANEMOI_BASE_SEED="${ANEMOI_BASE_SEED:-12345}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

echo "Running verification export"
echo "  checkpoint: $CHECKPOINT_PATH"
echo "  start/end : $START -> $END"
echo "  frequency : $FREQ"
echo "  dataset   : $DATASET_PATH"
echo "  run root  : $RUN_ROOT"

CMD=(
  anemoi-training train
  --config-path "$CONFIG_PATH"
  --config-name "$CONFIG_NAME"
  system.input.warm_start="$CHECKPOINT_PATH"
  system.input.dataset="$DATASET_PATH"
  system.output.root="$RUN_ROOT"
  dataloader.training.datasets.data.start="$START"
  dataloader.training.datasets.data.end="$END"
  dataloader.validation.datasets.data.start="$START"
  dataloader.validation.datasets.data.end="$END"
  dataloader.test.datasets.data.start="$START"
  dataloader.test.datasets.data.end="$END"
  data.frequency="$FREQ"
  dataloader.training.datasets.data.frequency="$FREQ"
  dataloader.validation.datasets.data.frequency="$FREQ"
  dataloader.test.datasets.data.frequency="$FREQ"
  training.multistep_input=2
  training.rollout.start=1
  training.rollout.max=1
  dataloader.validation_rollout=1
  training.num_sanity_val_steps=0
  diagnostics.export_predictions.enabled=true
  diagnostics.export_predictions.format=netcdf
  diagnostics.export_predictions.every_n_batches=1
  diagnostics.export_predictions.parameters='[height_500,refc,temp_500,ugrd_500,vgrd_500]'
)

if [[ -n "$GRAPH_PATH" ]]; then
  CMD+=(system.input.graph="$GRAPH_PATH")
fi

"${CMD[@]}"

if [[ ! -d "$EXPORT_DIR" ]]; then
  echo "ERROR: export directory not found: $EXPORT_DIR"
  exit 3
fi

echo "Generating per-valid-time frames (1h lead) into: $FRAMES_ROOT"

python - "$EXPORT_DIR" "$FRAMES_ROOT" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

export_dir = Path(sys.argv[1])
frames_root = Path(sys.argv[2])

files = sorted(export_dir.glob("pred_target_epoch*_batch*.nc"))
if not files:
    raise SystemExit(f"No export files found in {export_dir}")

var_map = [
    ("height_500", "hgt500"),
    ("refc", "refc"),
    ("temp_500", "temperature500"),
    ("ugrd_500", "u500"),
    ("vgrd_500", "v500"),
]

seen_valid = set()
frame_idx = 0
written = {alias: 0 for _, alias in var_map}
missing = set()

for path in files:
    ds = xr.open_dataset(path)
    if "variable" not in ds.coords:
        continue

    variables = [str(v) for v in ds.coords["variable"].values]

    if ds.sizes.get("target_time", 0) < 1 or ds.sizes.get("pred_time", 0) < 1:
        continue

    valid_np = np.datetime64(ds.coords["target_time"].values[0])
    valid_key = str(valid_np)
    if valid_key in seen_valid:
        continue
    seen_valid.add(valid_key)

    # YYYYMMDDTHHMMSS for filenames
    valid_s = np.datetime_as_string(valid_np, unit="s").replace("-", "").replace(":", "")

    lat = ds.coords["latitude"].values if "latitude" in ds.coords else None
    lon = ds.coords["longitude"].values if "longitude" in ds.coords else None

    for var_name, alias in var_map:
        if var_name not in variables:
            missing.add(var_name)
            continue

        i = variables.index(var_name)
        targ = ds["target"].isel(target_time=0).values[:, i]
        pred = ds["prediction"].isel(pred_time=0).values[:, i]
        err = targ - pred

        out_dir = frames_root / alias
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{alias}_lead01_valid_{valid_s}_frame_{frame_idx:05d}.png"

        fig, axs = plt.subplots(1, 3, figsize=(15, 5), layout="tight")
        titles = [
            f"truth {alias}\\nvalid {np.datetime_as_string(valid_np, unit='s')}",
            f"prediction {alias} (lead 1h)\\nvalid {np.datetime_as_string(valid_np, unit='s')}",
            f"error truth-pred {alias}\\nvalid {np.datetime_as_string(valid_np, unit='s')}",
        ]

        if lat is None or lon is None:
            x = np.arange(targ.shape[0])
            axs[0].plot(x, targ, linewidth=0.5)
            axs[1].plot(x, pred, linewidth=0.5)
            axs[2].plot(x, err, linewidth=0.5)
            for ax, title in zip(axs, titles):
                ax.set_title(title)
                ax.set_xlabel("node")
        else:
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

        fig.savefig(out_file, dpi=150)
        plt.close(fig)
        written[alias] += 1

    frame_idx += 1

manifest = frames_root / "frames_manifest.txt"
with manifest.open("w", encoding="utf-8") as f:
    f.write("# variable_alias frame_count\n")
    for _, alias in var_map:
        f.write(f"{alias} {written[alias]}\n")

print(f"Wrote manifest: {manifest}")
for _, alias in var_map:
    print(f"{alias}: {written[alias]} frames")
if missing:
    print("Missing variables in export (skipped):", ", ".join(sorted(missing)))

print("\nVideo example per variable:")
print("  ffmpeg -framerate 6 -pattern_type glob -i 'video_frames/hgt500/*_frame_*.png' -c:v libx264 -pix_fmt yuv420p hgt500.mp4")
PY

echo "Done."
echo "Frames root: $FRAMES_ROOT"
echo "Manifest  : $FRAMES_ROOT/frames_manifest.txt"
