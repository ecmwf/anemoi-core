#!/bin/bash -l
#SBATCH -J DE371_diffusion
#SBATCH -A p200177
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --qos=default

set -x
module load env/staging/2024.1
module load NVHPC
module load GCC
module load Python/3.11.10-GCCcore-13.3.0
load_puv

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH=/home/users/u101957/code/anemoi-core/models/src/anemoi/models:$PYTHONPATH
cd /home/users/u102751/code/anemoi/anemoi-core/training/src/anemoi/training/config

# Important: on force la visibilité "0,1,2,3" (recommandé dans l'exemple MeluXina)
CUDA_VISIBLE_DEVICES=0,1,2,3 srun --ntasks=1 \
  puv run torchrun --standalone --nproc_per_node=4 \
  -m anemoi-training train --config-name=config_angelique config-dir /home/users/u102751/code/anemoi/anemoi-core/training/src/anemoi/training/config