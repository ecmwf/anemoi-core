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
#SBATCH -o anemoi-gpu-1mon-6hrleadtime_123hr_loss-verify.%j.out
#SBATCH -e anemoi-gpu-1mon-6hrleadtime-123hr_loss-verify.%j.err

set -euo pipefail

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate anemoi-training-env-python3.12
cd /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core


#clt run_id="8f3510f7-a380-46f0-a670-6d5a341c2882"
#cltrun_id="1fdd1f40-f4f7-4dca-9a69-c9a16f2459d5"
#run_id="7be32a16-2f83-479c-8370-1dc631dbbf81"
#run_id="7be32a16-2f83-479c-8370-1dc631dbbf81"
#run_id="58a03191-bccf-45c6-9b3c-19b314d1fbe7"
#run_id="5cdef9b4-a3fa-4105-8a31-edfbe0cc0c79"
run_id="5cdef9b4-a3fa-4105-8a31-edfbe0cc0c79"
training/docs/user-guide/examples/run_rrfs_verify_export_1to6h.sh \
  /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/checkpoint/${run_id}/inference-last.ckpt \
  2024-05-11T09:00:00 2024-05-31T20:00:00 1h



# dd  then read PNGs with out01 in filenames ( that is +2h lead when n_step_output=1). 
