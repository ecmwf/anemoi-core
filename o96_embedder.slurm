#!/bin/bash
#SBATCH --job-name=o96-embedder
#SBATCH --account=deste_340_26
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

cd /leonardo_work/DestE_340_26/users/jschrei1/anemoi-core
source venv/bin/activate

export MLFLOW_ALLOW_FILE_STORE=true

anemoi-training train +experiment=o96_era5_transformer
