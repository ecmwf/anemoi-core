#!/usr/bin/env bash
# Usage:
#   run_rrfs_verify.sh <checkpoint_path> <start> <end> <frequency> [graph_path]
# Example:
#   run_rrfs_verify.sh /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/checkpoint/<run_id>/last.ckpt \
#     2024-05-05T00:00:00 2024-05-05T23:00:00 1h
# Optional graph override:
#   run_rrfs_verify.sh /path/to/ckpt 2024-05-05T00:00:00 2024-05-05T23:00:00 1h \
#     /scratch3/.../graphs/rrfs-3km-lam-graph-20km-fresh.pt
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: run_rrfs_verify.sh <checkpoint_path> <start> <end> <frequency> [graph_path]"
  echo "Example:"
  echo "  run_rrfs_verify.sh /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/checkpoint/<run_id>/last.ckpt \\"
  echo "    2024-05-05T00:00:00 2024-05-05T23:00:00 1h"
  exit 1
fi

CHECKPOINT_PATH="$1"
START="$2"
END="$3"
FREQ="$4"
GRAPH_PATH="${5:-}"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "ERROR: checkpoint file not found: $CHECKPOINT_PATH"
  exit 2
fi

if [[ -n "$GRAPH_PATH" && ! -f "$GRAPH_PATH" ]]; then
  echo "ERROR: graph file not found: $GRAPH_PATH"
  echo "Note: this verify config uses graph=existing, so the .pt file must already exist."
  exit 2
fi

export ANEMOI_BASE_SEED="${ANEMOI_BASE_SEED:-12345}"

CMD=(
  anemoi-training train
  --config-path /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core/training/docs/user-guide/examples \
  --config-name anemoi-training-rrfs-lam-neural-lam-verify \
  system.input.warm_start="$CHECKPOINT_PATH" 
  dataloader.training.datasets.data.start="$START" 
  dataloader.training.datasets.data.end="$END" 
  dataloader.validation.datasets.data.start="$START" 
  dataloader.validation.datasets.data.end="$END" 
  dataloader.test.datasets.data.start="$START" 
  dataloader.test.datasets.data.end="$END" 
  data.frequency="$FREQ"
)

if [[ -n "$GRAPH_PATH" ]]; then
  CMD+=(system.input.graph="$GRAPH_PATH")
fi

"${CMD[@]}"
