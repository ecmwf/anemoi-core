#!/bin/bash
#SBATCH --qos=ng
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --account=ecaifs  # ecrdasca  # ecaifs
#SBATCH --mem=220G
#SBATCH --time=48:00:00
#SBATCH --output=outputs/single01-%j.out
#set -x

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
# export CUDA_LAUNCH_BLOCKING=1
# export HYDRA_FULL_ERROR=1
# export NCCL_DEBUG=TRACE
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=ib0,lo

# generic settings

export ANEMOI_BASE_SEED=30963664

GITDIR=/perm/daep/projects/anemoi-core/feature-ailand/training/config-ailand
WORKDIR=$GITDIR
cd $WORKDIR

export CUDA_LAUNCH_BLOCKING=1

# generic settings
VENV=/home/daep/PERM/projects/anemoi-core/feature-ailand/training/venv

module load python3
source ${VENV}/bin/activate

#srun anemoi-training train --config-name=single
#srun anemoi-training train --config-name=single_snow
#srun anemoi-training train --config-name=single_small
srun anemoi-training train --config-name=ailand training.run_id=55032c61586543d0930a0544c3f946c4
