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

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from anemoi.training.train.train import AnemoiTrainer
from anemoi.training.utils.config import load_config
from anemoi.utils.mlflow.client import AnemoiMlflowClient

os.environ["ANEMOI_BASE_SEED"] = "42"  # need to set base seed if running on github runners


LOGGER = logging.getLogger(__name__)


@pytest.mark.mlflow
@pytest.mark.slow
def test_config_build(tmp_path: Path, mlflow_server: str) -> None:

    # Load without resolving: interpolations such as
    # system.input.graph = ${system.output.root}/... depend on system.output.root,
    # which is only set below. Resolving eagerly would collapse them to MISSING.
    config = load_config(
        "training/tests/integration/config/atmo_integration_test.yaml",
        resolve=False,
    )

    config.system.output.root = str(tmp_path)
    # Log to the CI MLflow instance (--mlflow-server) the runner is logged into,
    # rather than the production server hardcoded in the config. tracking_uri is a
    # literal string, so it is safe to override before resolving interpolations.
    config.diagnostics.log.mlflow.tracking_uri = mlflow_server
    OmegaConf.resolve(config)

    assert config.diagnostics.log.interval == 50

    trainer = AnemoiTrainer(config)
    trainer.train()

    client = AnemoiMlflowClient(mlflow_server, authentication=True)

    reference_id: Final = "684814cd86ed45f383bed3f0a87a782b"
    metric: Final = "train_multi_dataset_loss_step"

    # Printed so a failing run's ID can be promoted to the new reference_id.
    LOGGER.info("Run ID from trainer: %s", trainer.run_id)
    print(f"Run ID from trainer: {trainer.run_id}")  # noqa: T201

    def get_loss_df(run_id: str) -> pd.DataFrame:
        history = client.get_metric_history(run_id, metric)
        return pd.DataFrame(
            {
                "step": [m.step for m in history],
                "loss": [m.value for m in history],
            },
        ).set_index("step")

    def is_similar(run_id1: str, run_id2: str) -> bool:
        df1, df2 = get_loss_df(run_id1), get_loss_df(run_id2)
        if df1.empty:
            msg = f"Run {run_id1} has no '{metric}' history on {mlflow_server}"
            raise ValueError(msg)
        if df2.empty:
            msg = f"Reference run {run_id2} has no '{metric}' history on {mlflow_server}"
            raise ValueError(msg)
        # Align on step before comparing: np.allclose compares positionally and
        # would silently broadcast-error (or mismatch) if the step sets differ.
        aligned = df1.join(df2, how="inner", lsuffix="_1", rsuffix="_2")
        if aligned.empty:
            msg = f"Runs {run_id1} and {run_id2} share no common steps"
            raise ValueError(msg)
        return np.allclose(aligned.loc[:, "loss_1"], aligned.loc[:, "loss_2"])

    assert is_similar(
        trainer.run_id,
        reference_id,
    ), f"Loss curve for run {trainer.run_id} does not match reference"
