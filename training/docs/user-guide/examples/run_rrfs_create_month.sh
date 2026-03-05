#!/usr/bin/env bash
set -euo pipefail

# Create one RRFS Zarr dataset (month-tagged output), then add boundary_mask and
# rewrite time coords.
#
# Usage:
#   run_rrfs_create_month.sh <YYYYMM> [output_dir] [recipe_yaml]
#
# Example:
#   run_rrfs_create_month.sh 202405
#   run_rrfs_create_month.sh 202405 /scratch3/NCEPDEV/fv3-cam/Ting.Lei/data
#
# Optional overrides (environment variables):
#   LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, BOUNDARY_KM, TIME_UNIT
# Defaults:
#   LAT_MIN=25, LAT_MAX=40, LON_MIN=-105, LON_MAX=-90, BOUNDARY_KM=20, TIME_UNIT=s

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: run_rrfs_create_month.sh <YYYYMM> [output_dir] [recipe_yaml]"
  exit 1
fi

YYYYMM="$1"
OUT_DIR="${2:-./tmp/rrfs-monthly}"
RECIPE="${3:-training/docs/user-guide/examples/anemoi-data-lam-rrfs-hres-example.yaml}"

if [[ ! "$YYYYMM" =~ ^[0-9]{6}$ ]]; then
  echo "ERROR: YYYYMM must be 6 digits, got: $YYYYMM"
  exit 2
fi
if [[ ! -f "$RECIPE" ]]; then
  echo "ERROR: recipe not found: $RECIPE"
  exit 2
fi

LAT_MIN="${LAT_MIN:-25}"
LAT_MAX="${LAT_MAX:-40}"
LON_MIN="${LON_MIN:--105}"
LON_MAX="${LON_MAX:--90}"
BOUNDARY_KM="${BOUNDARY_KM:-20}"
TIME_UNIT="${TIME_UNIT:-s}"

NAME_TAG="$YYYYMM"

mkdir -p "$OUT_DIR" "./tmp"
TMP_RECIPE="./tmp/anemoi-data-rrfs-${NAME_TAG}.yaml"
RAW_ZARR="${OUT_DIR}/rrfs-conus-3km-${NAME_TAG}-raw.zarr"
FINAL_ZARR="${OUT_DIR}/rrfs-conus-3km-${NAME_TAG}-bcmask-time-${TIME_UNIT}.zarr"

# Build a month-tagged recipe copy (date range stays exactly as in the source recipe).
cp "$RECIPE" "$TMP_RECIPE"
sed -i -E "s|^name:.*$|name: rrfs-conus-3km-${NAME_TAG}-1h|g" "$TMP_RECIPE"

START="$(sed -n -E 's/^[[:space:]]*start:[[:space:]]*"?([^"]+)"?[[:space:]]*$/\1/p' "$TMP_RECIPE" | head -n1)"
END="$(sed -n -E 's/^[[:space:]]*end:[[:space:]]*"?([^"]+)"?[[:space:]]*$/\1/p' "$TMP_RECIPE" | head -n1)"
FREQ="$(sed -n -E 's/^[[:space:]]*frequency:[[:space:]]*"?([^"]+)"?[[:space:]]*$/\1/p' "$TMP_RECIPE" | head -n1)"

if [[ -z "${START}" || -z "${END}" || -z "${FREQ}" ]]; then
  echo "ERROR: Could not parse dates.start/end/frequency from $TMP_RECIPE"
  echo "Please ensure they are set under 'dates:'."
  exit 3
fi

echo "Creating monthly dataset:"
echo "  month:      $YYYYMM"
echo "  source yaml: $RECIPE"
echo "  start/end:  $START -> $END"
echo "  frequency:  $FREQ"
echo "  recipe:     $TMP_RECIPE"
echo "  raw:        $RAW_ZARR"
echo "  final:      $FINAL_ZARR"

anemoi-datasets create "$TMP_RECIPE" "$RAW_ZARR" --overwrite

python training/docs/user-guide/examples/add_boundary_mask.py \
  "$RAW_ZARR" \
  "$FINAL_ZARR" \
  --lon-min "$LON_MIN" \
  --lon-max "$LON_MAX" \
  --lat-min "$LAT_MIN" \
  --lat-max "$LAT_MAX" \
  --boundary-km "$BOUNDARY_KM" \
  --var-name boundary_mask \
  --start "$START" \
  --frequency "$FREQ" \
  --time-unit "$TIME_UNIT"

echo "Done: $FINAL_ZARR"
