#!/bin/bash
#SBATCH -A fv3-cam
#SBATCH -J anemoi-train
#SBATCH -p u1-h100
#SBATCH -q gpuwf
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 8:00:00
#SBATCH -o anemoi-gpu-1mon-6hrleadtime_123hr_loss-no_refc_input-verify.%j.out
#SBATCH -e anemoi-gpu-1mon-6hrleadtime-123hr_loss-no_refc_input-verify.%j.err

set -euo pipefail

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate anemoi-training-env-python3.12
cd /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core

run_id="95f6db58-d330-481c-8b1c-0eb58f39c621"
training/docs/user-guide/examples/run_rrfs_verify_export_1to6h_no_refc_input.sh \
  /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/checkpoint/${run_id}/inference-last.ckpt \
  2024-05-11T09:00:00 2024-05-31T20:00:00 1h
