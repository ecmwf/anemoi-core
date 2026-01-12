#!/usr/bin/env python3

import subprocess
import logging
from pathlib import Path
import os
from datetime import datetime

from anemoi.training.run_manage.slurm_job import get_date_folder, SlurmJob


hres = "o1280"

TRAINING_ITERATIONS = 6e5
CONFIG = f"hindcast_{hres}"
RUN_NAME = f"{hres}_{TRAINING_ITERATIONS:.0e}_{get_date_folder()}"
max_steps = 72
STEPS = list(range(0, max_steps + 1, 12))
STEPS_STR = str(STEPS).replace(" ", "")
RESIDUAL_STATISTICS = f"{hres}_dict_0_72.npy"

# gpus
NUM_GPUS_PER_MODEL = 4
SLURM_NODES = "1"
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
    ],
}


# ATOS - conda
HPC = os.environ["HPC"]

SLURM_CONFIG = {
    "nodes": SLURM_NODES,
    "time": "01:00:00",
    "gpus-per-node": SLURM_GPUS_PER_NODE,
    "ntasks-per-node": SLURM_NTASKS_PER_NODE,
    "cpus-per-task": "8",
    "mem": "256G",
    "output": Path.home().joinpath(
        "dev/jobscripts/outputs",
        get_date_folder(),
        TRAINING_PARAMS["config"] + "-%j.out",
    ),
    "qos": "ng",
}


GIT_REPOS = {
    "anemoi-training": "ds-diffusion",
    "anemoi-models": "ds-diffusion",
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
