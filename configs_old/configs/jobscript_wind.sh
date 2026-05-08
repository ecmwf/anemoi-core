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

export PYTHONUNBUFFERED=1

source /e/data1/jureap-data/ecmwf/users/clare1/time_interpolator_env/bin/activate

srun anemoi-training train dataloader=native_grid_nowind training.fork_run_id=f791b1b356e44c3cbf4e53e45ec086c2 diagnostics.log.mlflow.run_name='no_wind_rain' --config-name=single_11_new_rollout
