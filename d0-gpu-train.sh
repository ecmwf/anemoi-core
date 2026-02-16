#!/bin/bash
#SBATCH -A fv3-cam
#SBATCH -J anemoi-train
#SBATCH -p u1-h100
#SBATCH -q gpuwf
#SBATCH -N 2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 24:00:00
#SBATCH -o anemoi-gpu-train.%j.out
#SBATCH -e anemoi-gpu-train.%j.err

set -euo pipefail

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate anemoi-training-env-python3.12
cd /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core

export ANEMOI_BASE_SEED=12345
export HYDRA_FULL_ERROR=1
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# base cache path
export TRITON_CACHE_DIR=/scratch3/NCEPDEV/fv3-cam/Ting.Lei/triton_cache
mkdir -p ${TRITON_CACHE_DIR}/${SLURM_JOB_ID}

echo "=== Anemoi GPU preflight ==="
echo "date: $(date)"
echo "host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_NNODES=${SLURM_NNODES:-}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-}"
echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
echo "override system.hardware.num_gpus_per_node=2"
echo "override system.hardware.num_nodes=2"
echo "override system.hardware.num_gpus_per_model=4"
echo "============================"

srun --gpu-bind=closest \
  --export=ALL,TRITON_CACHE_DIR=/scratch3/NCEPDEV/fv3-cam/Ting.Lei/triton_cache/${SLURM_JOB_ID}/${SLURM_PROCID} \
  anemoi-training train \
    --config-path /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core/training/docs/user-guide/examples \
    --config-name anemoi-training-rrfs-lam-neural-lam-static-forcing \
    system.hardware.num_gpus_per_node=2 \
    system.hardware.num_nodes=2 \
    system.hardware.num_gpus_per_model=4
