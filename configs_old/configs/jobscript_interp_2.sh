#!/bin/bash
#SBATCH --job-name=anemoi-jupiter-example
#SBATCH --nodes=8
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

srun anemoi-training train training.run_id=95d57ac1a4be4bae8c67e3b1dfe33558 diagnostics.log.mlflow.run_name='graph 6 interp diff' --config-name=diff_interp_only
