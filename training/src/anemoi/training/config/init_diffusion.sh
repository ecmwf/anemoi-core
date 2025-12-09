#!/bin/bash -l
#SBATCH -J DE371_diffusion
#SBATCH -A p200177
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -p gpu
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --qos=short

# Environment loading
module load python3/3.11.10-01
load_puv

# Debugging
export ANEMOI_BASE_SEED=277
export HYDRA_FULL_ERROR=1
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_CUDA_DSA=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# echo "Running on nodes: $SLURM_NODELIST"


# === TRAINING CONFIG ===
# CONFIG_NAME="diffusion_test_incond"
# CONFIG_PATH="/home/users/u101957/code/anemoi-core/training/src/anemoi/training/config/"
# OUTPUT_DIR="/home/users/u101957/code/anemoi-core/training/src/anemoi/training/config/output/diffusion_test_incond/"


# === LAUNCH TRAINING ===
srun puv run anemoi-training train --config-name=diffusion_incond.yaml \

echo "Script execution completed: Successful run"
