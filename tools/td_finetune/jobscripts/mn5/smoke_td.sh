#!/bin/bash
# 1 node, debug QoS: (1) CPU check that the warm-start checkpoint loads and equals the converted weights,
# (2) 30 training steps on 4 GPUs (2 ensemble groups) to check memory/speed.
#SBATCH --job-name=td-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --time=00:40:00
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
echo "== load check (CPU) $(date)"
python check_load.py 2>&1 | grep -v "triton\|torch-cluster\|pip install\|TORCH_VERSION\|NVML\|warnings.warn\|^$\|Installing\|install it"
echo "== smoke train $(date)"
srun anemoi-training train --config-dir=$TD/configs --config-name=td_no_w_mn5 \
    system.hardware.num_nodes=1 training.max_steps=30 \
    dataloader.limit_batches.training=30 dataloader.limit_batches.validation=2 \
    diagnostics.log.mlflow.enabled=false
echo "== done $(date)"
