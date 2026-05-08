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

srun anemoi-training train training.run_id=a044a3d47e8846e88040890a2cc26b4f training.scalers.general_variable.weights.cp=0.1 training.scalers.general_variable.weights.tp=1 training.load_weights_only=False training.transfer_learning=False diagnostics.log.mlflow.run_name="high cp tp 1" --config-name=fine_ensinterp_agg
