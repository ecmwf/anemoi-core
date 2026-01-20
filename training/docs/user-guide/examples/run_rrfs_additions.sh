#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: run_rrfs_additions.sh <additions_yaml> <existing_zarr>"
  exit 1
fi

ADD_YAML="$1"
ZARR_PATH="$2"

anemoi-datasets init-additions "$ADD_YAML" "$ZARR_PATH"
anemoi-datasets load-additions "$ADD_YAML" "$ZARR_PATH"
anemoi-datasets finalise-additions "$ADD_YAML" "$ZARR_PATH"

echo "Additions complete for $ZARR_PATH"
