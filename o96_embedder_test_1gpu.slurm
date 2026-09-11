#!/bin/bash
#SBATCH --job-name=o96-embedder-1gpu-test
#SBATCH --account=<TU_ACCOUNT>          # p.ej. DestE_340_26 -- pon el tuyo real
#SBATCH --partition=<TU_PARTITION>      # p.ej. boost_usr_prod -- pon el tuyo real
#SBATCH --qos=<TU_QOS>                  # si tu cuenta lo requiere, si no borra esta linea
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

cd /leonardo_work/DestE_340_26/users/jschrei1/anemoi-core
source venv/bin/activate

export MLFLOW_ALLOW_FILE_STORE=true

anemoi-training train +experiment=o96_era5_transformer \
  system.hardware.num_gpus_per_node=1
