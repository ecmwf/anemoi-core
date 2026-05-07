#!/usr/bin/env bash
# Run direct +1h verification for checkpoints trained with refc as
# diagnostic/output-only, refc value/range-weighted loss, and finer_graph_v1.
# Usage:
#   run_rrfs_verify_export_1h_refc_value_no_refc_input_finer_graph_v1.sh <checkpoint_path> <start> <end> <frequency>
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: run_rrfs_verify_export_1h_refc_value_no_refc_input_finer_graph_v1.sh <checkpoint_path> <start> <end> <frequency>"
  exit 1
fi

CHECKPOINT_PATH="$1"
START="$2"
END="$3"
FREQ="$4"
CONFIG_NAME="anemoi-training-rrfs-lam-neural-lam-verify-202405-1h-refc-value-loss-no-refc-input-finer-graph-v1"
RESOLVED_CONFIG="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tmp-verify-resolved-1h-refc-value-no-refc-input-finer-graph-v1.yaml"

echo "DEBUG_CONFIG_NAME: $CONFIG_NAME"
echo "DEBUG_VERIFY_ROOT: /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/verify/finer_graph_v1_1h_refc_value_no_refc_input/"

export ANEMOI_BASE_SEED="${ANEMOI_BASE_SEED:-12345}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export ANEMOI_LOG_LEVEL="${ANEMOI_LOG_LEVEL:-DEBUG}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

python - <<PY
from anemoi.datasets import open_dataset
path = "/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core/tmp/rrfs-monthly/rrfs-conus-3km-202405-bcmask-time-s.zarr"
ds = open_dataset(path, start="${START}", end="${END}", frequency="${FREQ}")
print("DEBUG_DATASET_PATH:", path)
print("DEBUG_DATASET_LEN:", len(ds.dates))
print("DEBUG_DATASET_START:", ds.dates[0])
print("DEBUG_DATASET_END:", ds.dates[-1])
PY

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
  --cfg job > "$RESOLVED_CONFIG"

echo "DEBUG_CONFIG_SAVED: $RESOLVED_CONFIG"

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

