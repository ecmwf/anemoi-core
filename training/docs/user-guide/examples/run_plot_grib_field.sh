#!/usr/bin/env bash
set -euo pipefail

# Example usage of plot_grib_field.py
#
# Adjust paths before running.
#
# Usage:
#   ./run_plot_grib_field.sh

GRIB_FILE="/path/to/rrfs.v2024050500.grib2"
OUT_PNG="t850.png"

python training/docs/user-guide/examples/plot_grib_field.py \
  "${GRIB_FILE}" \
  --shortName t \
  --typeOfLevel isobaricInhPa \
  --level 850 \
  --lat-min 25 --lat-max 40 --lon-min -105 --lon-max -90 \
  --out "${OUT_PNG}"

echo "Wrote ${OUT_PNG}"
