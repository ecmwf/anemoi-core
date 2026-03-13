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
#SBATCH -t 24:00:00
#SBATCH -o anemoi-gpu-1mon-verify.%j.out
#SBATCH -e anemoi-gpu-1mon-verify.%j.err

set -euo pipefail

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate anemoi-training-env-python3.12
cd /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core


run_id="9ca82919-27de-4c6d-8edb-f441052164d1"
bash training/docs/user-guide/examples/run_ffs_verification_202505.sh \
  /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/checkpoint/${run_id}/inference-last.ckpt \
  2024-05-02T09:00:00 2024-05-31T20:00:00 1h

