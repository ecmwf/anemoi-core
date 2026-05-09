#!/bin/bash
#SBATCH -A fv3-cam
#SBATCH -J anemoi-verify-base
#SBATCH -p u1-h100
#SBATCH -q gpuwf
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 8:00:00
#SBATCH -o anemoi-gpu-1mon-1hrleadtime-1hr_refc_value-base-refc_input-no_hydrometeors-finer_graph_v1-single_input-verify.%j.out
#SBATCH -e anemoi-gpu-1mon-1hrleadtime-1hr_refc_value-base-refc_input-no_hydrometeors-finer_graph_v1-single_input-verify.%j.err

set -euo pipefail

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate anemoi-training-env-python3.12
cd /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core

run_id="REPLACE_WITH_RUN_ID"
training/docs/user-guide/examples/run_rrfs_verify_export_1h_refc_value_base_refc_input_no_hydrometeors_finer_graph_v1_single_input.sh \
  /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/base_graphtransformer_finer_graph_v1_single_input_refc_input_no_hydrometeors/checkpoint/${run_id}/inference-last.ckpt \
  2024-05-02T09:00:00 2024-05-30T23:00:00 1h
