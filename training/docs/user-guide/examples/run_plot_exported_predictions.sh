#!/usr/bin/env bash
set -euo pipefail

# Example usage of plot_exported_predictions.py
#
# Adjust paths before running.
#
# Usage:
#   ./run_plot_exported_predictions.sh

EXPORT_NC="/path/to/pred_target_epoch000_batch0000.nc"
DATASET_ZARR="/path/to/test-20km-bcmask-time.zarr"
VAR_NAME="temp_850"
TIME_INDEX=0
OUT_PNG="temp_850_t0.png"

python training/docs/user-guide/examples/plot_exported_predictions.py \
  "${EXPORT_NC}" \
  --dataset "${DATASET_ZARR}" \
  --variable "${VAR_NAME}" \
  --time-index "${TIME_INDEX}" \
  --out "${OUT_PNG}"

echo "Wrote ${OUT_PNG}"
