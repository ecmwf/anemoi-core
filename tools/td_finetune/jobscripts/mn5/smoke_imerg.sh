#!/bin/bash
# 1 node, debug QoS: real 5k_lr5e-6 config, 30 steps, 2 validation batches, to check checkpoint load, the 6-term loss and speed.
#SBATCH --job-name=td-imerg-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --time=00:50:00
#SBATCH --exclusive
#SBATCH -p acc
#SBATCH -A ehpc01
#SBATCH --qos=acc_debug
#SBATCH -o /gpfs/scratch/ehpc01/ecm807005/td_mn5/logs/%x-%j.out

source /gpfs/scratch/ehpc01/ecm807005/td_mn5/env_td.sh
export SLURM_GPUS_PER_NODE=4
export AIFS_NUM_GPUS_PER_MODEL=1
export ANEMOI_BASE_SEED=50046951
cd $TD
srun --kill-on-bad-exit=1 anemoi-training train --config-dir=$TD/configs --config-name=temporal_downscaler_imerg_ft_5k_lr5e-6 \
    system.hardware.num_nodes=1 training.max_steps=30 \
    dataloader.limit_batches.training=30 dataloader.limit_batches.validation=2 \
    diagnostics.log.mlflow.enabled=false system.output.root=$TD/out_smoke \
    system.output.checkpoints.root=$TD/out_smoke/checkpoint system.output.plots=$TD/out_smoke/plots
echo "== done $(date)"
