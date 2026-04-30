#!/bin/bash
#SBATCH -A fv3-cam
#SBATCH -J anemoi-loss-table
#SBATCH -p u1-h100
#SBATCH -q gpuwf
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 8:00:00
#SBATCH -o anemoi-gpu-1mon-6hrleadtime_123hr_refc_value_loss-no_refc_input-loss_table.%j.out
#SBATCH -e anemoi-gpu-1mon-6hrleadtime_123hr_refc_value_loss-no_refc_input-loss_table.%j.err

set -euo pipefail

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate anemoi-training-env-python3.12
cd /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core

run_id="CHANGE_ME_REFC_VALUE_NO_REFC_INPUT_RUN_ID"
training/docs/user-guide/examples/run_rrfs_verify_loss_table_1to6h_refc_value_no_refc_input.sh \
  /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/checkpoint/${run_id}/inference-last.ckpt \
  2024-05-05T00:00:00 2024-05-30T23:00:00 1h
