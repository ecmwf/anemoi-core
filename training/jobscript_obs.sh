#!/bin/bash

#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --account=ecaifs
#SBATCH --output=outputs/aifsobs_train.out.%j

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=ib0,lo

# generic settings
CONDA_ENV=aifs-dev

module load conda
conda activate $CONDA_ENV
echo "Running training job on $(hostname) at $(pwd)"
srun anemoi-training train hardware=slurm --config-name=obs
