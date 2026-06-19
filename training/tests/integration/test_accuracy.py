# (C) Copyright 2025 Anemoi contributors.
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
import pandas as pd
import numpy as np


from anemoi.training.train.train import AnemoiTrainer

from anemoi.training.utils.config import load_config

os.environ["ANEMOI_BASE_SEED"] = "42"  # need to set base seed if running on github runners


LOGGER = logging.getLogger(__name__)


# @skip_if_offline
# @pytest.mark.mlflow
# @pytest.mark.slow
def test_config_build(tmp_path: Path) -> None:

    # assert Path.cwd() == 5

    config = load_config("training/tests/integration/config/atmo_integration_test.yaml")

    # TODO double check
    config.system.output.root = str(tmp_path)

    assert config.diagnostics.log.interval == 50

    trainer = AnemoiTrainer(config)
    assert trainer.cfg.diagnostics.log.interval == 50

    trainer.train()

    client = AnemoiMlflowClient("https://mlflow.ecmwf.int/", authentication=True)
    experiment = client.get_experiment_by_name("aifs-convergence")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=10,
    )

    reference_id: Final = "e00340e8cd5c41d2881afd2265677321"

    def get_loss_df(run_id: str) -> pd.DataFrame:
        history = client.get_metric_history(
            run_id,
            "train_multi_dataset_loss_step",
        )
        return pd.DataFrame(
            {
                "step": [m.step for m in history],
                "loss": [m.value for m in history],
            },
        ).set_index("step")

    def is_similar(run_id1: str, run_id2: str) -> bool:
        df1, df2 = get_loss_df(run_id1), get_loss_df(run_id2)
        return np.allclose(df1.loc[:, "loss"], df2.loc[:, "loss"])

    assert is_similar(
        runs[0].info.run_id, reference_id,
    ), f"Loss curve for run {runs[0].info.run_id} does not match reference"
