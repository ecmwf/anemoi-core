#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash training/docs/user-guide/examples/run_rrfs_lam_inference.sh \
#     <checkpoint> <date> <lead_time> [output_nc] [device]
#
# Example:
#   bash training/docs/user-guide/examples/run_rrfs_lam_inference.sh \
#     /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/checkpoint/<run_id>/inference-last.ckpt \
#     2024-05-20T00:00:00 24h \
#     /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/infer/rrfs_lam_20240520.nc \
#     cpu

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Usage: run_rrfs_lam_inference.sh <checkpoint> <date> <lead_time> [output_nc] [device]"
  exit 1
fi

CKPT="$1"
DATE="$2"
LEAD="$3"
OUT_NC="${4:-/tmp/rrfs_lam_inference.nc}"
DEVICE="${5:-cpu}"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT"
  exit 2
fi

CFG="training/docs/user-guide/examples/anemoi-inference-rrfs-lam.yaml"
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: config not found: $CFG"
  exit 2
fi

mkdir -p "$(dirname "$OUT_NC")"

echo "Checking checkpoint supporting arrays for output_mask (controls lateral boundary forcing)..."
set +e
anemoi-inference metadata --supporting-arrays "$CKPT" | grep -E "output_mask|grid_indices|cutout_mask"
STATUS=$?
set -e
if [[ $STATUS -ne 0 ]]; then
  echo "WARN: no output_mask-like array printed; boundary forcing may be disabled."
fi

echo "Running inference..."
anemoi-inference run "$CFG" \
  checkpoint="$CKPT" \
  date="$DATE" \
  lead_time="$LEAD" \
  output.netcdf.path="$OUT_NC" \
  device="$DEVICE"

echo "Done: $OUT_NC"
