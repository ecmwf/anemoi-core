#!/usr/bin/env bash
set -euo pipefail

# Example runner for node-coverage plot.
#
# Edit paths before running.

EXPORT_FILE="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/verify/predictions/pred_target_epoch000_batch0000.nc"
DATASET_ZARR="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core/test-20km-bcmask-time-s.zarr"
OUT_PNG="model_vs_data_nodes.png"

python training/docs/user-guide/examples/plot_model_vs_data_nodes.py \
  --export "${EXPORT_FILE}" \
  --dataset "${DATASET_ZARR}" \
  --out "${OUT_PNG}"

echo "Wrote ${OUT_PNG}"

