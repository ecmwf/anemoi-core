#!/bin/bash -l
#SBATCH -J DE371_diffusion
#SBATCH -A p200177
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -p gpu
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --qos=default


export TORCH_DISTRIBUTED_DEBUG=INFO 
export OMP_NUM_THREADS=4
export CUDA_HOME=/usr/local/cuda-12.1
export NVHPC_CUDA_HOME=/usr/local/cuda-12.1
export CXX=g++ #the compiler for cpp extensions
export CC=gcc  #the compiler to access the good cpp standard
export NCCL_ASYNC_ERROR_HANDLING=1
module load env/release/2024.1
module load env/staging/2024.1
module load NVHPC
module load GCC
module load Python/3.11.10-GCCcore-13.3.0
load_puv



# Echo des commandes lancees
set -x

cd /home/users/u102751/code/anemoi/anemoi-core/training
export HYDRA_FULL_ERROR=1

srun puv run anemoi-training train --config-name=diffusion_test