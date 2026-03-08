#!/bin/bash
#SBATCH -A fv3-cam
#SBATCH -J anemoi-train-gh200
#SBATCH -p u1-gh
#SBATCH -q gpuwf
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 24:00:00
#SBATCH -o anemoi-gh200-train.%j.out
#SBATCH -e anemoi-gh200-train.%j.err

set -euo pipefail

CONDA_EXE=/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/conda
cd /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core

export ANEMOI_BASE_SEED=12345
export HYDRA_FULL_ERROR=1
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TRITON_CACHE_DIR=/scratch3/NCEPDEV/fv3-cam/Ting.Lei/triton_cache/${SLURM_JOB_ID}
mkdir -p "$TRITON_CACHE_DIR"

"$CONDA_EXE" --version

srun --gpu-bind=closest \
  --export=ALL,TRITON_CACHE_DIR=/scratch3/NCEPDEV/fv3-cam/Ting.Lei/triton_cache/${SLURM_JOB_ID}/${SLURM_PROCID} \
  "$CONDA_EXE" run -n anemoi-training-env-python3.12 anemoi-training train \
    --config-path /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core/training/docs/user-guide/examples \
    --config-name anemoi-training-rrfs-lam-neural-lam-static-forcing \
    system.hardware.num_gpus_per_node=1 \
    system.hardware.num_nodes=2 \
    system.hardware.num_gpus_per_model=2
