#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=/home/ecm5702/dev/jobscripts/inference/outputs/%j.out
#SBATCH --qos=ng

set -eux
cd /home/ecm5702/dev/jobscripts/inference

source /home/ecm5702/hpcperm/.ds/bin/activate
export ANEMOI_BASE_SEED=756
export HYDRA_FULLL_ERROR=1
export NCCL_IB_TIMEOUT=30
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEV=/home/ecm5702/dev/
export DATA_DIR=/home/mlx/ai-ml/datasets/
export DATA_STABLE_DIR=/home/mlx/ai-ml/datasets/stable/
export OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/
export INTER_MAT_DIR=/ec/res4/hpcperm/nesl/inter_mat/
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
export HPC="atos"

inference="/home/ecm5702/dev/anemoi-training/src/anemoi/training/diagnostics/local_inference/save_sampling.py"
srun --export=ALL,HPC python $inference --name_exp 8c8d95213c8e4df6b5784795cd6411d2/ --N_members 3 --nsteps=20 --N_samples 2
# f0f35da002d042f98ec60cfcda614815
