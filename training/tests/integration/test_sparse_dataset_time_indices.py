# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytest_mock import MockFixture

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule
from anemoi.training.tasks import SparseForecaster


class FakeDatasetReader:
    def __init__(self, *, dataset_name: str, frequency: str, start: str, stop: str, step_minutes: int) -> None:
        self.data = dataset_name
        self.frequency = frequency
        self.dates = np.arange(
            np.datetime64(start),
            np.datetime64(stop),
            np.timedelta64(step_minutes, "m"),
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


def build_sparse_forecaster_config() -> DictConfig:
    return OmegaConf.create(
        {
            "data": {
                "timestep": "5m",
                "datasets": {
                    "meps": {"forcing": ["forcing_var"], "diagnostic": [], "target": []},
                    "nordic_radar": {"forcing": ["forcing_var"], "diagnostic": [], "target": []},
                },
            },
            "task": {
                "_target_": "anemoi.training.tasks.SparseForecaster",
                "multistep_input": 1,
                "multistep_output": 1,
                "timestep": "5m",
                "rollout": {"start": 1, "epoch_increment": 0, "maximum": 3},
                "validation_rollout": 1,
            },
            "dataloader": {
                "pin_memory": False,
                "prefetch_factor": 2,
                "debug": {"time_index_mode": "auto_sparse"},
                "num_workers": {"training": 0, "validation": 0, "test": 0},
                "batch_size": {"training": 1, "validation": 1, "test": 1},
                "training": {
                    "datasets": {
                        "meps": {
                            "dataset_config": {"dataset": "meps_source", "frequency": "1h"},
                            "start": "2020-01-01 00:00:00",
                            "end": "2020-01-02 00:00:00",
                        },
                        "nordic_radar": {
                            "dataset_config": {"dataset": "radar_source", "frequency": "5m"},
                            "start": "2020-01-01 00:00:00",
                            "end": "2020-01-02 00:00:00",
                        },
                    },
                },
                "validation": {"datasets": {}},
                "test": {"datasets": {}},
            },
            "training": {},
        },
    )


def test_sparse_forecaster_metadata_keeps_per_dataset_sparse_windows(mocker: MockFixture) -> None:
    cfg = build_sparse_forecaster_config()
    fake_readers = {
        "meps_source": FakeDatasetReader(
            dataset_name="meps_source",
            frequency="1h",
            start="2020-01-01T00:00",
            stop="2020-01-03T00:00",
            step_minutes=60,
        ),
        "radar_source": FakeDatasetReader(
            dataset_name="radar_source",
            frequency="5m",
            start="2020-01-01T00:00",
            stop="2020-01-03T00:00",
            step_minutes=5,
        ),
    }

    mocker.patch(
        "anemoi.training.data.datamodule.create_dataset",
        side_effect=lambda dataset_cfg, **_kwargs: fake_readers[dataset_cfg.dataset_config.dataset],
    )

    task = SparseForecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="5m",
        rollout={"start": 1, "epoch_increment": 0, "maximum": 3},
    )
    datamodule = AnemoiDatasetsDataModule(config=cfg, task=task)
    metadata = {"metadata_inference": {}}
    datamodule.fill_metadata(metadata)
    task.fill_metadata(metadata)

    timesteps = metadata["metadata_inference"]["nordic_radar"]["timesteps"]
    assert timesteps["relative_date_indices_training"] == [0, 1, 2, 3]
    assert timesteps["relative_date_indices_training_by_dataset"]["meps"] == [0]
    assert timesteps["relative_date_indices_training_by_dataset"]["nordic_radar"] == [0, 1, 2, 3]
