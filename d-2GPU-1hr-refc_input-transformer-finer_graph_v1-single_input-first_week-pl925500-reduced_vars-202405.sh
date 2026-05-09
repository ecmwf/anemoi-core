#!/bin/bash
#SBATCH -A fv3-cam
#SBATCH -J anemoi-rvtr
#SBATCH -p u1-h100
#SBATCH -q gpuwf
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 24:00:00
#SBATCH -o anemoi-gpu-2gpu-1hr_refc_input-transformer-finer_graph_v1-single_input-first_week-pl925500-reduced_vars-train.%j.out
#SBATCH -e anemoi-gpu-2gpu-1hr_refc_input-transformer-finer_graph_v1-single_input-first_week-pl925500-reduced_vars-train.%j.err

set -euo pipefail

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate anemoi-training-env-python3.12
cd /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core

export ANEMOI_BASE_SEED=12345
export HYDRA_FULL_ERROR=1
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TRITON_CACHE_DIR=/scratch3/NCEPDEV/fv3-cam/Ting.Lei/triton_cache/${SLURM_JOB_ID}
mkdir -p "$TRITON_CACHE_DIR"

GRAPH_BASE=/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/graphs/rrfs-3km-lam-graph-finer_graph_v1.pt
GRAPH_DATA=/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/graphs/rrfs-3km-lam-graph-finer_graph_v1_data.pt
if [[ ! -e "$GRAPH_DATA" ]]; then
  ln -s "$GRAPH_BASE" "$GRAPH_DATA"
fi
ls -l "$GRAPH_BASE" "$GRAPH_DATA"

srun --gpu-bind=closest \
  --export=ALL,TRITON_CACHE_DIR=/scratch3/NCEPDEV/fv3-cam/Ting.Lei/triton_cache/${SLURM_JOB_ID}/${SLURM_PROCID} \
  anemoi-training train \
    --config-path /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core/training/docs/user-guide/examples \
    --config-name anemoi-training-rrfs-lam-neural-lam-static-forcing-202405-1h-refc-input-transformer-finer-graph-v1-single-input-first-week-pl925500-reduced-vars \
    system.hardware.num_gpus_per_node=2 \
    system.hardware.num_nodes=1 \
    system.hardware.num_gpus_per_model=1
