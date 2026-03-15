#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash training/docs/user-guide/examples/run_plot_zarr_refc_max_timeseries.sh \
#     <dataset_zarr> <start> <end> [out_dir]
#
# Example:
#   bash training/docs/user-guide/examples/run_plot_zarr_refc_max_timeseries.sh \
#     /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core/tmp/rrfs-monthly/rrfs-conus-3km-202405-bcmask-time-s.zarr \
#     2024-05-02T09:00:00 2024-05-31T20:00:00 \
#     /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/verify/refc_timeseries

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: run_plot_zarr_refc_max_timeseries.sh <dataset_zarr> <start> <end> [out_dir]"
  exit 1
fi

DATASET="$1"
START="$2"
END="$3"
OUT_DIR="${4:-.}"

mkdir -p "$OUT_DIR"

python training/docs/user-guide/examples/plot_zarr_refc_max_timeseries.py \
  "$DATASET" \
  --variable refc \
  --percentile 95 \
  --start "$START" \
  --end "$END" \
  --out-png "$OUT_DIR/refc_max_timeseries.png" \
  --out-csv "$OUT_DIR/refc_max_timeseries.csv"

echo "Wrote:"
echo "  $OUT_DIR/refc_max_timeseries.png"
echo "  $OUT_DIR/refc_max_timeseries.csv"
