#!/usr/bin/env bash
set -euo pipefail

#if [[ $# -ne 1 ]]; then
#  echo "Usage: run_rrfs_additions.sh <additions_yaml>"
#  exit 1
#fi

ADD_YAML="./training/docs/user-guide/examples/anemoi-data-rrfs-additions.yaml"
ZARR_PATH="test-20km-bcmask.zarr"

anemoi-datasets init-additions "$ADD_YAML"
anemoi-datasets load-additions "$ADD_YAML"
anemoi-datasets finalise-additions "$ZARR_PATH"

echo "Additions complete for $ZARR_PATH"
