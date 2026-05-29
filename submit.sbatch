#!/bin/bash --login
#SBATCH --job-name=rollout-testing
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --hint=nomultithread
#SBATCH --time=2:59:00
#SBATCH --account=ecaifs
#SBATCH --qos=ng
#SBATCH -o %x-%j.out


export SLURM_GPUS_PER_NODE=2
export SLURM_NNODES=1

source .venv/bin/activate
cd .training

srun anemoi-training train
