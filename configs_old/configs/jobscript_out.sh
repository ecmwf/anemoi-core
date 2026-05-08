#!/bin/bash
#SBATCH --job-name=anemoi-jupiter-example
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --account=gkpdm
#SBATCH -p booster
#SBATCH --time=12:00:00
#SBATCH -o %x-%j.out

#source /e/data1/jureap-data/ecmwf/users/clare1/time_interpolator_env/bin/activate

module load Stages/2025  GCCcore/.13.3.0
module load Python/3.12.3

source /e/data1/jureap-data/ecmwf/users/clare1/time_interpolator_env/bin/activate

srun anemoi-training train training.run_id=5994b5275fc5458cb977bf99bcec6bae --config-name=single_11_new_rollout
