#!/bin/bash -l
#SBATCH -J DE371_diffusion
#SBATCH -A p200177
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=10:00:00
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
echo $CUDA_VISIBLE_DEVICES
cd /home/users/u102751/code/anemoi/anemoi-env/
puv run /home/users/u102751/code/anemoi/create_mean_tensor.py