#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=494000
#SBATCH --output=/leonardo/home/userexternal/jdumontl/dev/jobscripts/inference/outputs/%j.out
#SBATCH --partition=boost_usr_prod
#SBATCH --account=DestE_340_25


 

set -eux
cd /leonardo/home/userexternal/jdumontl/dev
source /leonardo/home/userexternal/jdumontl/dev/aifs/bin/activate
export ANEMOI_BASE_SEED=756
export HYDRA_FULLL_ERROR=1
export NCCL_IB_TIMEOUT=30
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEV=/leonardo/home/userexternal/jdumontl/dev/
export DATA_DIR=/leonardo_work/DestE_340_25/ai-ml/datasets/
export DATA_STABLE_DIR=/leonardo_work/DestE_340_25/ai-ml/datasets/
export OUTPUT=/leonardo_work/DestE_340_25/output/jdumontl/downscaling/
export GRID_DIR=/leonardo_work/DestE_340_25/AIFS_grids
export INTER_MAT_DIR=/leonardo/home/userexternal/jdumontl/inter_mat/
export RESIDUAL_STATISTICS_DIR=/leonardo/home/userexternal/jdumontl/residuals_statistics/
export HPC="leo"
inference="/leonardo/home/userexternal/jdumontl/dev/anemoi-training/src/anemoi/training/diagnostics/local_inference/save_sampling.py"
srun --export=ALL,HPC python $inference --name_exp 5cfa2a40fa214167847fb1d9b5161812 --N_members 3 --use_ds_predict True
# db341ebef7fd40a782dea123c40f7a06 f3f95029fb2e4f80a58929e9f841e34d 9624b40f93144a7486d3f692f3d23680