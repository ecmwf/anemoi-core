#!/bin/bash
#SBATCH -A fv3-cam
#SBATCH -J anemoi-verify-1h-fgv1
#SBATCH -p u1-h100
#SBATCH -q gpuwf
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 8:00:00
#SBATCH -o anemoi-gpu-1mon-1hrleadtime-1hr_refc_value_loss-no_refc_input-finer_graph_v1-verify.%j.out
#SBATCH -e anemoi-gpu-1mon-1hrleadtime-1hr_refc_value_loss-no_refc_input-finer_graph_v1-verify.%j.err

set -euo pipefail

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate anemoi-training-env-python3.12
cd /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core

run_id="dfe60ea5-25f4-4c32-957a-81b58d203c26"
training/docs/user-guide/examples/run_rrfs_verify_export_1h_refc_value_no_refc_input_finer_graph_v1.sh \
  /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/checkpoint/${run_id}/inference-last.ckpt \
  2024-05-05T00:00:00 2024-05-30T23:00:00 1h
