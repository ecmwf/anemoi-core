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
from omegaconf import OmegaConf
from pytest_mock import MockFixture

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule
from anemoi.training.tasks import Forecaster
from anemoi.utils.dates import frequency_to_seconds
from anemoi.utils.dates import frequency_to_timedelta


class FakeDatasetReader:
    def __init__(self, *, dataset_name: str, frequency: str, start: str, stop: str) -> None:
        self.data = dataset_name
        self.frequency = frequency_to_timedelta(frequency)
        self.dates = np.arange(
            np.datetime64(start),
            np.datetime64(stop),
            np.timedelta64(int(frequency_to_timedelta(frequency).total_seconds() // 60), "m"),
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
        msg = "FakeDatasetReader is only used for datamodule timing tests."
        raise NotImplementedError(msg)


def get_reader_dataset_config(dataset_cfg: Any) -> dict[str, Any]:
    if hasattr(dataset_cfg, "dataset_config"):
        return dict(dataset_cfg.dataset_config)
    if isinstance(dataset_cfg, dict):
        return dict(dataset_cfg["dataset_config"])
    msg = f"Unsupported dataset config type: {type(dataset_cfg)!r}"
    raise TypeError(msg)


def make_multidataset_cfg(*, meps_frequency: str, radar_frequency: str, data_frequency: str = "5m") -> Any:
    return OmegaConf.create(
        {
            "data": {
                "frequency": data_frequency,
                "datasets": {
                    "meps": {"forcing": ["forcing_var"], "diagnostic": [], "target": []},
                    "nordic_radar": {"forcing": ["forcing_var"], "diagnostic": [], "target": []},
                },
            },
            "task": {
                "_target_": "anemoi.training.tasks.Forecaster",
                "multistep_input": 1,
                "multistep_output": 1,
                "timestep": "5m",
                "rollout": {"start": 1, "epoch_increment": 0, "maximum": 1},
                "validation_rollout": 1,
            },
            "dataloader": {
                "pin_memory": False,
                "training": {
                    "datasets": {
                        "meps": {
                            "dataset_config": {"dataset": "meps_source", "frequency": meps_frequency},
                            "end": "2020-01-02",
                        },
                        "nordic_radar": {
                            "dataset_config": {"dataset": "radar_source", "frequency": radar_frequency},
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


def test_datamodule_relative_date_indices_follow_task_config_for_mixed_frequency_forecaster(
    mocker: MockFixture,
) -> None:
    cfg = OmegaConf.create(
        {
            "data": {
                "frequency": "5m",
                "datasets": {
                    "meps": {"forcing": ["forcing_var"], "diagnostic": [], "target": []},
                    "nordic_radar": {"forcing": ["forcing_var"], "diagnostic": [], "target": []},
                },
            },
            "task": {
                "_target_": "anemoi.training.tasks.Forecaster",
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

    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="5m",
        rollout={"start": 1, "epoch_increment": 0, "maximum": 3},
    )
    mocker.patch(
        "anemoi.training.data.datamodule.create_dataset",
        side_effect=lambda dataset_cfg, **_kwargs: FakeDatasetReader(
            dataset_name=dataset_cfg.dataset_config.dataset,
            frequency=dataset_cfg.dataset_config.frequency,
            start="2020-01-01T00:00",
            stop="2020-01-03T00:00",
        ),
    )
    datamodule = AnemoiDatasetsDataModule(config=cfg, task=task)

    assert datamodule.ds_train.model_relative_date_indices.tolist() == [0, 1, 2, 3]


def test_datamodule_mixed_frequency_alignment_uses_task_timestep_without_data_frequency(
    mocker: MockFixture,
) -> None:
    cfg = OmegaConf.create(
        {
            "data": {
                "datasets": {
                    "meps": {"forcing": [], "diagnostic": [], "target": []},
                    "radar": {"forcing": [], "diagnostic": [], "target": []},
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
                        "meps": {"dataset_config": {"dataset": "meps_source", "frequency": "1h"}, "end": "2020-01-02"},
                        "radar": {
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

    task = Forecaster(
        multistep_input=2,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 1, "epoch_increment": 0, "maximum": 1},
    )
    mocker.patch(
        "anemoi.training.data.datamodule.create_dataset",
        side_effect=lambda dataset_cfg, **_kwargs: FakeDatasetReader(
            dataset_name=dataset_cfg.dataset_config.dataset,
            frequency=dataset_cfg.dataset_config.frequency,
            start="2020-01-01T00:00",
            stop="2020-01-03T00:00",
        ),
    )
    datamodule = AnemoiDatasetsDataModule(config=cfg, task=task)

    assert datamodule.ds_train.model_relative_date_indices.tolist() == [0, 1, 2]


def test_datamodule_keeps_dense_path_for_aligned_frequencies(
    mocker: MockFixture,
) -> None:
    cfg = make_multidataset_cfg(meps_frequency="5m", radar_frequency="5m")
    created_frequencies: list[str] = []

    def _create_dataset(dataset_cfg: Any, **_kwargs: Any) -> FakeDatasetReader:
        dataset_config = get_reader_dataset_config(dataset_cfg)
        frequency = dataset_config.get("interpolate_frequency", dataset_config.get("frequency"))
        created_frequencies.append(frequency)
        return FakeDatasetReader(
            dataset_name=dataset_config["dataset"],
            frequency=frequency,
            start="2020-01-01T00:00",
            stop="2020-01-03T00:00",
        )

    mocker.patch("anemoi.training.data.datamodule.create_dataset", side_effect=_create_dataset)
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="5m",
        rollout={"start": 1, "epoch_increment": 0, "maximum": 1},
    )
    datamodule = AnemoiDatasetsDataModule(config=cfg, task=task)

    ds_train = datamodule.ds_train

    assert created_frequencies == ["5m", "5m"]
    assert ds_train.relative_date_indices_are_native


def test_datamodule_uses_mixed_frequency_alignment_for_mixed_frequencies(
    mocker: MockFixture,
) -> None:
    cfg = make_multidataset_cfg(meps_frequency="1h", radar_frequency="5m")
    created_configs: list[dict[str, Any]] = []

    def _create_dataset(dataset_cfg: Any, **_kwargs: Any) -> FakeDatasetReader:
        dataset_config = get_reader_dataset_config(dataset_cfg)
        created_configs.append(dataset_config)
        frequency = dataset_config.get("interpolate_frequency", dataset_config.get("frequency"))
        return FakeDatasetReader(
            dataset_name=dataset_config["dataset"],
            frequency=frequency,
            start="2020-01-01T00:00",
            stop="2020-01-03T00:00",
        )

    mocker.patch("anemoi.training.data.datamodule.create_dataset", side_effect=_create_dataset)
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="5m",
        rollout={"start": 1, "epoch_increment": 0, "maximum": 1},
    )
    datamodule = AnemoiDatasetsDataModule(config=cfg, task=task)

    ds_train = datamodule.ds_train

    assert created_configs == [
        {"dataset": "meps_source", "frequency": "1h"},
        {"dataset": "radar_source", "frequency": "5m"},
    ]
    assert not ds_train.relative_date_indices_are_native
    assert ds_train._anchor_dataset_name == "nordic_radar"
    assert ds_train.model_relative_date_indices.tolist() == [0, 1]
    assert frequency_to_seconds(ds_train.data_readers["meps"].frequency) == frequency_to_seconds("1h")
    assert frequency_to_seconds(ds_train.data_readers["nordic_radar"].frequency) == frequency_to_seconds("5m")


def test_datamodule_mixed_frequency_alignment_prefers_task_timestep_over_data_frequency(
    mocker: MockFixture,
) -> None:
    cfg = make_multidataset_cfg(meps_frequency="1h", radar_frequency="5m", data_frequency="1h")

    mocker.patch(
        "anemoi.training.data.datamodule.create_dataset",
        side_effect=lambda dataset_cfg, **_kwargs: FakeDatasetReader(
            dataset_name=dataset_cfg.dataset_config.dataset,
            frequency=dataset_cfg.dataset_config.frequency,
            start="2020-01-01T00:00",
            stop="2020-01-03T00:00",
        ),
    )
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="5m",
        rollout={"start": 1, "epoch_increment": 0, "maximum": 1},
    )
    datamodule = AnemoiDatasetsDataModule(config=cfg, task=task)

    ds_train = datamodule.ds_train

    assert ds_train.model_relative_date_indices.tolist() == [0, 1]
    assert ds_train.data_relative_date_indices_by_dataset["meps"].tolist() == [0]
    assert ds_train.data_relative_date_indices_by_dataset["nordic_radar"].tolist() == [0, 1]


def test_datamodule_fill_metadata_derives_mixed_frequency_windows_from_task(
    mocker: MockFixture,
) -> None:
    cfg = make_multidataset_cfg(meps_frequency="1h", radar_frequency="5m")

    mocker.patch(
        "anemoi.training.data.datamodule.create_dataset",
        side_effect=lambda dataset_cfg, **_kwargs: FakeDatasetReader(
            dataset_name=dataset_cfg.dataset_config.dataset,
            frequency=dataset_cfg.dataset_config.frequency,
            start="2020-01-01T00:00",
            stop="2020-01-03T00:00",
        ),
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

    timesteps = metadata["metadata_inference"]["nordic_radar"]["timesteps"]

    assert timesteps["relative_date_indices_training"] == [0, 1, 2, 3]
    assert timesteps["relative_date_input_indices_training_by_dataset"] == {
        "meps": [0],
        "nordic_radar": [0],
    }
    assert timesteps["relative_date_target_indices_training_by_dataset"] == {
        "meps": [],
        "nordic_radar": [1, 2, 3],
    }


def test_datamodule_and_task_metadata_configure_mixed_frequency_timing(
    mocker: MockFixture,
) -> None:
    cfg = make_multidataset_cfg(meps_frequency="1h", radar_frequency="5m")

    mocker.patch(
        "anemoi.training.data.datamodule.create_dataset",
        side_effect=lambda dataset_cfg, **_kwargs: FakeDatasetReader(
            dataset_name=dataset_cfg.dataset_config.dataset,
            frequency=dataset_cfg.dataset_config.frequency,
            start="2020-01-01T00:00",
            stop="2020-01-03T00:00",
        ),
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
    task.configure_from_metadata(metadata)

    timesteps = metadata["metadata_inference"]["nordic_radar"]["timesteps"]
    assert timesteps["relative_date_indices_training"] == [0, 1, 2, 3]
    assert timesteps["relative_date_input_indices_training_by_dataset"]["meps"] == [0]
    assert timesteps["relative_date_input_indices_training_by_dataset"]["nordic_radar"] == [0]
    assert timesteps["relative_date_indices_training_by_dataset"]["meps"] == [0]
    assert timesteps["relative_date_indices_training_by_dataset"]["nordic_radar"] == [0, 1, 2, 3]
    assert timesteps["relative_date_target_indices_training_by_dataset"]["meps"] == []
    assert timesteps["relative_date_target_indices_training_by_dataset"]["nordic_radar"] == [1, 2, 3]
    assert task.dataset_time_maps["meps"] == {0: 0}
    assert task.dataset_time_maps["nordic_radar"] == {0: 0, 1: 1, 2: 2, 3: 3}
