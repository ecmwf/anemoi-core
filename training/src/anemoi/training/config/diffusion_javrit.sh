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
echo $CUDA_VISIBLE_DEVICES
cd /home/users/u102751/code/anemoi/anemoi-env/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
srun puv run anemoi-training train --config-name=config_dev_slurm --config-dir /home/users/u102751/code/anemoi/anemoi-core/training/src/anemoi/training/config