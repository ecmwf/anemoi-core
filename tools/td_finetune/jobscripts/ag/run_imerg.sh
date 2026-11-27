#!/bin/bash --login
# IMERG fine-tune of the TD no_w checkpoint on ag (GH200). Usage (always 8 nodes: the configs' lr assumes 8 x 4 GPUs):
#   sbatch --job-name=td-imerg-5k-lr5e-6 run_imerg.sh temporal_downscaler_imerg_ft_5k_lr5e-6
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --hint=nomultithread
#SBATCH --time=16:00:00
#SBATCH --account=ecaifs
#SBATCH -p gpu
#SBATCH -o /scratch/momc/td_ag/logs/%x-%j.out

source /scratch/momc/td_ag/env_td.sh
CONFIG=${1:?config name}
shift
if [ "$SLURM_NNODES" != "8" ] && [ -z "$ALLOW_NODES" ]; then
  echo "ERROR: effective lr in the config assumes 8 nodes, got $SLURM_NNODES (set ALLOW_NODES=1 to override)"; exit 1
fi
export SLURM_GPUS_PER_NODE=4
export AIFS_NUM_GPUS_PER_MODEL=1
export ANEMOI_BASE_SEED=${ANEMOI_BASE_SEED:-50046951}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-72}
ulimit -c 0
cd $TD
echo "$CONFIG: nodes=$SLURM_NNODES job=$SLURM_JOB_ID host=$(hostname) $(date)"
srun --kill-on-bad-exit=1 anemoi-training train --config-dir=$TD/configs --config-name=$CONFIG \
    system.hardware.num_nodes=$SLURM_NNODES "$@"
