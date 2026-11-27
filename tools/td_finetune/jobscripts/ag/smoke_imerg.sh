#!/bin/bash --login
# 1 node, gpu_debug: real 5k_lr5e-6 config, 30 steps, 2 validation batches (checkpoint load, 6-term loss, memory, speed).
#SBATCH --job-name=td-imerg-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --hint=nomultithread
#SBATCH --time=00:50:00
#SBATCH --account=ecaifs
#SBATCH -p gpu_debug
#SBATCH --qos=dg
#SBATCH -o /scratch/momc/td_ag/logs/%x-%j.out

source /scratch/momc/td_ag/env_td.sh
export SLURM_GPUS_PER_NODE=4
export AIFS_NUM_GPUS_PER_MODEL=1
export ANEMOI_BASE_SEED=50046951
ulimit -c 0
cd $TD
srun --kill-on-bad-exit=1 anemoi-training train --config-dir=$TD/configs --config-name=temporal_downscaler_imerg_ft_5k_lr5e-6 \
    system.hardware.num_nodes=1 training.max_steps=30 \
    dataloader.limit_batches.training=30 dataloader.limit_batches.validation=2 \
    diagnostics.log.mlflow.enabled=false system.output.root=$TD/out_smoke \
    system.output.checkpoints.root=$TD/out_smoke/checkpoint system.output.plots=$TD/out_smoke/plots
echo "== done $(date)"
