#!/bin/bash
# Temporal-downscaler training on MN5 (ACC). Usage:
#   sbatch -N 8 run_td.sh [hydra overrides...]
# Defaults: warm start from the converted no_w weights (see README_mn5.md). Extra args are passed to anemoi-training.
#SBATCH --job-name=td-no-w
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --time=71:00:00
#SBATCH --exclusive
#SBATCH -p acc
#SBATCH -A ehpc01
#SBATCH --qos=acc_ehpc
#SBATCH -o /gpfs/scratch/ehpc01/ecm807005/td_mn5/logs/%x-%j.out

source /gpfs/scratch/ehpc01/ecm807005/td_mn5/env_td.sh
export SLURM_GPUS_PER_NODE=4
export AIFS_NUM_GPUS_PER_MODEL=1
export ANEMOI_BASE_SEED=${ANEMOI_BASE_SEED:-50046951}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-20}
cd $TD
echo "td-no-w: nodes=$SLURM_NNODES job=$SLURM_JOB_ID host=$(hostname) $(date)"
srun anemoi-training train --config-dir=$TD/configs --config-name=td_no_w_mn5 \
    system.hardware.num_nodes=$SLURM_NNODES "$@"
