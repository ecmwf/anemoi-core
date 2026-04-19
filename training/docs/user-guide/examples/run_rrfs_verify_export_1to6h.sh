#!/usr/bin/env bash
# Run 1h-to-3h verification with export enabled.
# Usage:
#   run_rrfs_verify_export_1to3h.sh <checkpoint_path> <start> <end> <frequency>
# Example:
#   run_rrfs_verify_export_1to3h.sh /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/checkpoint/<run_id>/inference-last.ckpt \
#     2024-05-05T00:00:00 2024-05-05T23:00:00 1h
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: run_rrfs_verify_export_1to3h.sh <checkpoint_path> <start> <end> <frequency>"
  exit 1
fi

CHECKPOINT_PATH="$1"
START="$2"
END="$3"
FREQ="$4"
CONFIG_NAME="anemoi-training-rrfs-lam-neural-lam-verify-202405-1to6h"

echo "DEBUG_CONFIG_NAME: $CONFIG_NAME"
echo "DEBUG_PLOTS_DIR: /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/verify/plots/plots"

export ANEMOI_BASE_SEED="${ANEMOI_BASE_SEED:-12345}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export ANEMOI_LOG_LEVEL="${ANEMOI_LOG_LEVEL:-DEBUG}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

# Debug: show the dataset dates as seen by anemoi.datasets (outside the training loader)
python - <<PY
from anemoi.datasets import open_dataset
path = "/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core/tmp/rrfs-monthly/rrfs-conus-3km-202405-bcmask-time-s.zarr"
ds = open_dataset(path, start="${START}", end="${END}", frequency="${FREQ}")
print("DEBUG_DATASET_PATH:", path)
print("DEBUG_DATASET_LEN:", len(ds.dates))
print("DEBUG_DATASET_START:", ds.dates[0])
print("DEBUG_DATASET_END:", ds.dates[-1])
PY

# Debug: dump the resolved config with overrides for inspection (no training run)
anemoi-training train \
  --config-path /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core/training/docs/user-guide/examples \
  --config-name "$CONFIG_NAME" \
  system.input.warm_start="$CHECKPOINT_PATH" \
  dataloader.training.datasets.data.start="$START" \
  dataloader.training.datasets.data.end="$END" \
  dataloader.validation.datasets.data.start="$START" \
  dataloader.validation.datasets.data.end="$END" \
  dataloader.test.datasets.data.start="$START" \
  dataloader.test.datasets.data.end="$END" \
  data.frequency="$FREQ" \
  dataloader.training.datasets.data.frequency="$FREQ" \
  dataloader.validation.datasets.data.frequency="$FREQ" \
  dataloader.test.datasets.data.frequency="$FREQ" \
  training.num_sanity_val_steps=0 \
  --cfg job > /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tmp-verify-resolved-1to6h.yaml

echo "DEBUG_CONFIG_SAVED: /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tmp-verify-resolved-1to6h.yaml"

# Actual verify run
anemoi-training train \
  --config-path /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core/training/docs/user-guide/examples \
  --config-name "$CONFIG_NAME" \
  system.input.warm_start="$CHECKPOINT_PATH" \
  dataloader.training.datasets.data.start="$START" \
  dataloader.training.datasets.data.end="$END" \
  dataloader.validation.datasets.data.start="$START" \
  dataloader.validation.datasets.data.end="$END" \
  dataloader.test.datasets.data.start="$START" \
  dataloader.test.datasets.data.end="$END" \
  data.frequency="$FREQ" \
  dataloader.training.datasets.data.frequency="$FREQ" \
  dataloader.validation.datasets.data.frequency="$FREQ" \
  dataloader.test.datasets.data.frequency="$FREQ" \
  training.num_sanity_val_steps=0
