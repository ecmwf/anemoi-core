#!/bin/bash
#SBATCH -A fv3-cam
#SBATCH -J anemoi-create-regrid 
#SBATCH -p u1-h100
#SBATCH -q  gpuwf         
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH -t 48:00:00
#SBATCH -o run_anemoi-create-regrid.%j.out
#SBATCH -e run_anemoi-create-regrid.%j.err


#!/usr/bin/env bash
# Convert observational NetCDF training data to the memmap layout Aardvark expects.
# Edit the paths below to your locations before running.

set -euo pipefail
source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-debug/src-jedi-bundle-only4mir.sh
source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-ecmwf-mir/mir/set_mir.sh
source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate anemoi-training-env-python3.12 

#!/usr/bin/env bash
# Prepare ERA5 memmaps and norms on the target grid.
# Edit paths and variable list to match what you downloaded and what your training expects.

set -euo pipefail
cd /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core
# anemoi-transform make-regrid-matrix src-example.grib2 rrfs-3km-subdomain-grid.nc /scratch3/NCEPDEV/fv3-cam/Ting.Lei/regrid/rrfs-to-latlon-3km.npz



anemoi-transform make-regrid-matrix \
  /scratch3/NCEPDEV/fv3-cam/Ting.Lei/rrfs-valid/rrfs.v2024050500.grib2 \
  rrfs-3km-subdomain-grid.nc \
  /scratch3/NCEPDEV/fv3-cam/Ting.Lei/regrid/rrfs-to-latlon-3km.npz


