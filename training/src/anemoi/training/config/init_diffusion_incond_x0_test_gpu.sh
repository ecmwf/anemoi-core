#!/bin/bash -l
#SBATCH -J DE371_diffusion
#SBATCH -A p200177
#SBATCH -N 2
#SBATCH -p gpu
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --qos=short
set -x
module load env/staging/2024.1
module load NVHPC
module load GCC
module load Python/3.11.10-GCCcore-13.3.0
load_puv

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH=/home/users/u101957/code/anemoi-core/models/src:/home/users/u101957/code/anemoi-core/training/src:$PYTHONPATH
echo $CUDA_VISIBLE_DEVICES
cd /home/users/u101957/code/anemoi-core/training/src/anemoi/training/config
CUDA_VISIBLE_DEVICES=0,1,2,3
srun puv run anemoi-training train --config-name=diffusion_incond 

#   puv run torchrun  \
#   --nnodes=2 \
#   --nproc_per_node=4 \
#   -m anemoi.training train --config-name=diffusion_incond
