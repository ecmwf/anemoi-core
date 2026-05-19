# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytest_mock import MockFixture

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule
from anemoi.training.tasks import Forecaster
from anemoi.utils.dates import frequency_to_timedelta


class FakeDatasetReader:
    def __init__(self, *, dataset_name: str, frequency: str, start: str, stop: str) -> None:
        self.data = dataset_name
        self.frequency = frequency_to_timedelta(frequency)
        self.dates = np.arange(
            np.datetime64(start),
            np.datetime64(stop),
            np.timedelta64(int(self.frequency.total_seconds()), "s"),
        )
        self.missing = set()
        self.has_trajectories = False
        self.statistics = {}
        self.metadata = {}
        self.supporting_arrays = {}
        self.variables = ["forcing_var", "prog_var"]
        self.name_to_index = {"forcing_var": 0, "prog_var": 1}
        self.resolution = "test"

    def get_sample(self, *args: Any, **kwargs: Any) -> None:
        msg = "FakeDatasetReader is only used for metadata-oriented integration coverage."
        raise NotImplementedError(msg)


def load_mixed_frequency_multidatasets_config() -> DictConfig:
    cfg = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_mixed_frequency_multidatasets.yaml")
    assert isinstance(cfg, DictConfig)
    return cfg


def test_mixed_frequency_forecaster_metadata_derives_per_dataset_time_windows(mocker: MockFixture) -> None:
    cfg = load_mixed_frequency_multidatasets_config()
    fake_readers = {
        "meps_source": FakeDatasetReader(
            dataset_name="meps_source",
            frequency="1h",
            start="2020-01-01T00:00",
            stop="2020-01-03T00:00",
        ),
        "radar_source": FakeDatasetReader(
            dataset_name="radar_source",
            frequency="5m",
            start="2020-01-01T00:00",
            stop="2020-01-03T00:00",
        ),
    }

    mocker.patch(
        "anemoi.training.data.datamodule.create_dataset",
        side_effect=lambda dataset_cfg, **_kwargs: fake_readers[dataset_cfg.dataset_config.dataset],
    )

    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="5m",
        rollout={"start": 3, "epoch_increment": 0, "maximum": 3},
    )
    datamodule = AnemoiDatasetsDataModule(config=cfg, task=task)
    metadata = {"metadata_inference": {}}
    datamodule.fill_metadata(metadata)
    task.fill_metadata(metadata)

    timesteps = metadata["metadata_inference"]["nordic_radar"]["timesteps"]
    assert timesteps["relative_date_indices_training"] == [0, 1, 2, 3]
    assert timesteps["relative_date_input_indices_training_by_dataset"]["meps"] == [0]
    assert timesteps["relative_date_input_indices_training_by_dataset"]["nordic_radar"] == [0]
    assert timesteps["relative_date_indices_training_by_dataset"]["meps"] == [0]
    assert timesteps["relative_date_indices_training_by_dataset"]["nordic_radar"] == [0, 1, 2, 3]
    assert timesteps["relative_date_target_indices_training_by_dataset"]["meps"] == []
    assert timesteps["relative_date_target_indices_training_by_dataset"]["nordic_radar"] == [1, 2, 3]
