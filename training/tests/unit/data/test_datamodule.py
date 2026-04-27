# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from omegaconf import OmegaConf

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule
from anemoi.training.tasks import Forecaster
from anemoi.training.tasks import SparseForecaster


def test_datamodule_relative_date_indices_follow_task_config_for_sparse_forecaster() -> None:
    cfg = OmegaConf.create(
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
                "training": {
                    "datasets": {
                        "meps": {"dataset_config": {"dataset": "meps_source", "frequency": "1h"}, "end": "2020-01-02"},
                        "nordic_radar": {
                            "dataset_config": {"dataset": "radar_source", "frequency": "5m"},
                            "end": "2020-01-02",
                        },
                    },
                },
                "validation": {"datasets": {}},
                "test": {"datasets": {}},
            },
            "training": {},
        },
    )

    task = SparseForecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="5m",
        rollout={"start": 1, "epoch_increment": 0, "maximum": 3},
    )
    datamodule = AnemoiDatasetsDataModule(config=cfg, task=task)

    assert datamodule.relative_date_indices() == [0, 1, 2, 3]


def test_datamodule_timestep_falls_back_to_task_when_data_timestep_is_missing() -> None:
    cfg = OmegaConf.create(
        {
            "data": {
                "frequency": "6h",
                "datasets": {
                    "data": {"forcing": [], "diagnostic": [], "target": []},
                },
            },
            "task": {
                "_target_": "anemoi.training.tasks.Forecaster",
                "multistep_input": 2,
                "multistep_output": 1,
                "timestep": "6h",
                "rollout": {"start": 1, "epoch_increment": 0, "maximum": 1},
                "validation_rollout": 1,
            },
            "dataloader": {
                "pin_memory": False,
                "training": {
                    "datasets": {
                        "data": {"dataset_config": {"dataset": "source", "frequency": "6h"}, "end": "2020-01-02"},
                    },
                },
                "validation": {"datasets": {}},
                "test": {"datasets": {}},
            },
            "training": {},
        },
    )

    task = Forecaster(
        multistep_input=2,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 1, "epoch_increment": 0, "maximum": 1},
    )
    datamodule = AnemoiDatasetsDataModule(config=cfg, task=task)

    assert datamodule.config_timestep == "6h"
    assert datamodule._lead_time_for_step(1) == "6h"
