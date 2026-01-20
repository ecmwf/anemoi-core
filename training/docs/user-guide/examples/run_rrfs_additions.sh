#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: run_rrfs_additions.sh <additions_yaml>"
  exit 1
fi

ADD_YAML="$1"

anemoi-datasets init-additions "$ADD_YAML"
anemoi-datasets load-additions "$ADD_YAML"
anemoi-datasets finalise-additions "$ADD_YAML"

echo "Additions complete for config: $ADD_YAML"
