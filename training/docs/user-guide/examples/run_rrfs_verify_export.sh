#!/usr/bin/env bash
# Run verification with export enabled and sufficient rollout/multistep to produce outputs.
# Usage:
#   run_rrfs_verify_export.sh <checkpoint_path> <start> <end> <frequency>
# Example:
#   run_rrfs_verify_export.sh /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/checkpoint/<run_id>/inference-last.ckpt \
#     2024-05-05T00:00:00 2024-05-05T23:00:00 1h
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: run_rrfs_verify_export.sh <checkpoint_path> <start> <end> <frequency>"
  exit 1
fi

CHECKPOINT_PATH="$1"
START="$2"
END="$3"
FREQ="$4"

export ANEMOI_BASE_SEED="${ANEMOI_BASE_SEED:-12345}"

anemoi-training train \
  --config-path /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core/training/docs/user-guide/examples \
  --config-name anemoi-training-rrfs-lam-neural-lam-verify \
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
  training.multistep_input=2 \
  training.rollout.start=2 \
  training.rollout.max=2 \
  dataloader.validation_rollout=2 \
  training.num_sanity_val_steps=0
