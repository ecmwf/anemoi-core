#!/usr/bin/env bash
# Usage:
#   run_rrfs_verify.sh <checkpoint_path> <start> <end> <frequency>
# Example:
#   run_rrfs_verify.sh /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/checkpoint/<run_id>/last.ckpt \
#     2024-05-05T00:00:00 2024-05-05T23:00:00 1h
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: run_rrfs_verify.sh <checkpoint_path> <start> <end> <frequency>"
  echo "Example:"
  echo "  run_rrfs_verify.sh /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/checkpoint/<run_id>/last.ckpt \\"
  echo "    2024-05-05T00:00:00 2024-05-05T23:00:00 1h"
  exit 1
fi

CHECKPOINT_PATH="$1"
START="$2"
END="$3"
FREQ="$4"

anemoi-training train \
  --config-path /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core/training/docs/user-guide/examples \
  --config-name anemoi-training-rrfs-lam-neural-lam-verify \
  system.input.warm_start="$CHECKPOINT_PATH" \
  dataloader.training.start="$START" \
  dataloader.training.end="$END" \
  dataloader.validation.start="$START" \
  dataloader.validation.end="$END" \
  dataloader.test.start="$START" \
  dataloader.test.end="$END" \
  data.frequency="$FREQ"
