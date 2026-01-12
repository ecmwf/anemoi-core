#!/usr/bin/env python3

import subprocess
import logging
from pathlib import Path
import os
from datetime import datetime

from anemoi.training.run_manage.slurm_job import get_date_folder, SlurmJob

hres = "o320"

TRAINING_ITERATIONS = 1e6
CONFIG = f"hindcast_{hres}_1encoder"
RUN_NAME = f"{hres}_1encoder_{TRAINING_ITERATIONS:.0e}_{get_date_folder()}"
max_steps = 72
STEPS = list(range(0, max_steps+1, 12))
STEPS_STR = str(STEPS).replace(" ", "")
RESIDUAL_STATISTICS = f"{hres}_dict_0_72.npy"
TRAINING_ITERATIONS = int(TRAINING_ITERATIONS)
TRAINING_APPROACH = "probabilistic_high_noise"
#DATASET_X = "downscaling-od-cf-enfh-0001-mars-o320-2003-2023-12h-v3.zarr"
#lres_gaussian_layer = False
#input_at_high_res = True
#lres_std_noise = 0.1


# gpus
NUM_GPUS_PER_MODEL = 2
SLURM_NODES = "2"
SLURM_GPUS_PER_NODE = "4"
SLURM_NTASKS_PER_NODE = "4"

TRAINING_PARAMS = {
    "config": CONFIG,
    "gpus": f"hardware.num_gpus_per_model={NUM_GPUS_PER_MODEL}",
    "run_name": f"diagnostics.log.mlflow.run_name={RUN_NAME}",
    "additional_params": [
        f"training.lr.iterations={TRAINING_ITERATIONS}",
        f"dataloader.steps={STEPS_STR}",
        f"hardware.files.residual_statistics={RESIDUAL_STATISTICS}",
        f"training.approach={TRAINING_APPROACH}",
        #f"hardware.files.dataset_x={DATASET_X}",
        #f"training.lres_gaussian_layer={lres_gaussian_layer}",
       # f"training.input_at_high_res={input_at_high_res}",
       # f"training.lres_std_noise={lres_std_noise}",

    ],
}


# ATOS - conda
HPC = os.environ["HPC"]

SLURM_CONFIG = {
    "nodes": SLURM_NODES,
    "time": "24:00:00",
    "gpus-per-node": SLURM_GPUS_PER_NODE,
    "ntasks-per-node": SLURM_NTASKS_PER_NODE,
    "cpus-per-task": "8",
    "mem": "494000",
    "output": Path.home().joinpath(
        "dev/jobscripts/outputs",
        get_date_folder(),
        TRAINING_PARAMS["config"] + "-%j.out",
    ),
    "partition": "boost_usr_prod",
    "account": "DestE_340_25",
}


GIT_REPOS = {
    "anemoi-training": "ds-global",
    "anemoi-models": "ds-global",
    "anemoi-datasets": "feature/fake-hindcasts",
}

PATHS = {
    "workdir": Path.home().joinpath("dev"),
    "venv_path": Path(os.environ["VENV"]),
}


def main():
    """Main entry point."""
    job = SlurmJob(
        SLURM_CONFIG,
        TRAINING_PARAMS,
        GIT_REPOS,
        PATHS,
    )

    job.submit()


if __name__ == "__main__":
    main()
