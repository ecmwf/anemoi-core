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
#SBATCH --time=2:00:00
#SBATCH -o %x-%j.out

 
module load Stages/2025  GCCcore/.13.3.0
module load Python/3.12.3

export PYTHONUNBUFFERED=1

source /e/data1/jureap-data/ecmwf/users/clare1/time_interpolator_new_env/bin/activate

srun anemoi-training train diagnostics.log.mlflow.run_name="debug_new_code" --config-name=debug_td
