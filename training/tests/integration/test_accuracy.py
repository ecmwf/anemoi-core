# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
from pathlib import Path
from typing import Final

os.environ["ANEMOI_BASE_SEED"] = "42"  # need to set base seed if running on github runners
# Required for deterministic cuBLAS (torch.use_deterministic_algorithms, enabled via
# config.training.deterministic below). Must be set before CUDA is initialised, i.e.
# before torch is imported transitively by the anemoi imports below.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import pandas as pd
import pytest
import torch
from hydra import compose
from hydra import initialize
from omegaconf import OmegaConf
from torch.testing import assert_close

from anemoi.training.train.train import AnemoiTrainer
from anemoi.utils.mlflow.client import AnemoiMlflowClient

# cuDNN autotuning picks different kernels per run; disable for reproducibility.
# Lightning's deterministic=True (set via config below) does not cover this.
torch.backends.cudnn.benchmark = False


LOGGER = logging.getLogger(__name__)


@pytest.mark.mlflow
@pytest.mark.slow
def test_accuracy(tmp_path: Path, mlflow_server: str) -> None:

    # compose leaves interpolations such as
    # system.input.graph = ${system.output.root}/... unresolved; they depend on
    # system.output.root, which is only set below, and we resolve afterwards.
    # config_path is relative to this file; the autouse set_working_directory fixture
    # (integration/conftest.py) chdirs to the repo root so Path.cwd() resolves the yaml.
    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_accuracy"):
        template = compose(config_name="config", overrides=["model=graphtransformer"])

    modifications = OmegaConf.load(
        Path.cwd() / "training/tests/integration/config/accuracy_testing/test_global.yaml",
    )
    config = OmegaConf.merge(template, modifications)

    config.system.output.root = str(tmp_path)
    config.diagnostics.log.mlflow.tracking_uri = mlflow_server
    # Drive pl.Trainer(deterministic=True), which sets torch.use_deterministic_algorithms
    # and cudnn.deterministic. Combined with the module-level CUBLAS/cudnn settings, this
    # makes the loss curve reproducible on identical hardware/CUDA versions.
    config.training.deterministic = True
    OmegaConf.resolve(config)

    assert config.diagnostics.log.interval == 50

    trainer = AnemoiTrainer(config)
    trainer.train()

    client = AnemoiMlflowClient(mlflow_server, authentication=True)

    reference_id: Final = "684814cd86ed45f383bed3f0a87a782b"
    metric: Final = "train_multi_dataset_loss_step"

    # Printed so a failing run's ID can be promoted to the new reference_id.
    LOGGER.info("Run ID from trainer: %s", trainer.run_id)

    def get_loss_df(run_id: str) -> pd.DataFrame:
        history = client.get_metric_history(run_id, metric)
        return pd.DataFrame(
            {
                "step": [m.step for m in history],
                "loss": [m.value for m in history],
            },
        ).set_index("step")

    def assert_similar(run_id1: str, run_id2: str) -> None:
        df1, df2 = get_loss_df(run_id1), get_loss_df(run_id2)
        if df1.empty:
            msg = f"Run {run_id1} has no '{metric}' history on {mlflow_server}"
            raise ValueError(msg)
        if df2.empty:
            msg = f"Reference run {run_id2} has no '{metric}' history on {mlflow_server}"
            raise ValueError(msg)
        aligned = df1.join(df2, how="inner", lsuffix="_1", rsuffix="_2")
        if aligned.empty:
            msg = f"Runs {run_id1} and {run_id2} share no common steps"
            raise ValueError(msg)
        # assert_close does not accept pandas Series; hand it the underlying arrays.
        assert_close(
            aligned.loc[:, "loss_1"].to_numpy(),
            aligned.loc[:, "loss_2"].to_numpy(),
            msg=lambda msg: f"Loss curve for run {trainer.run_id} does not match reference\n{msg}",
        )

    assert_similar(trainer.run_id, reference_id)
