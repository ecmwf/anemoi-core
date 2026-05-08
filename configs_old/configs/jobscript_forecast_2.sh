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

srun anemoi-training train training.run_id=9890414472674d5c84cc964bf2318c7d training.scalers.general_variable.weights.cp=0.025 training.scalers.general_variable.weights.tp=0.25 training.load_weights_only=False training.transfer_learning=False diagnostics.log.mlflow.run_name="high cp tp 0.25" --config-name=fine_ensinterp_agg
