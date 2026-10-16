# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import contextlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytest_mock import MockFixture
from torch.utils.data import Dataset

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule
from anemoi.training.tasks import Forecaster
from anemoi.training.tasks import OffsetForecaster
from anemoi.training.tasks import TemporalDownscaler
from anemoi.training.tasks.base import BaseTask
from anemoi.training.utils import seeding
from anemoi.training.utils.seeding import SeedContext
from anemoi.training.utils.seeding import derive_seed
from anemoi.utils.dates import frequency_to_seconds
from anemoi.utils.dates import frequency_to_timedelta


class SampleIndexDataset(Dataset):
    """Returns the index of each sample, so batches show which samples were loaded."""

    def __init__(self, size: int = 1) -> None:
        self.size = size
        self.data_readers: dict = {}
        self.relative_date_indices_are_native = True

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index

    def set_relative_date_indices(self, relative_date_indices: dict) -> None:
        del relative_date_indices


def _fixed_rollout_task() -> Forecaster:
    return Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 1, "epoch_increment": 0, "maximum": 1},
    )


@pytest.fixture(autouse=True)
def _base_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the base seed, as AnemoiTrainer does before the datamodule builds its samplers."""
    monkeypatch.setattr(seeding, "_base_seed", 1000)


def _make_datamodule(
    task: BaseTask,
    *,
    persistent_workers: bool = False,
    batch_size: int = 1,
    num_workers: int = 1,
) -> AnemoiDatasetsDataModule:
    datamodule = AnemoiDatasetsDataModule.__new__(AnemoiDatasetsDataModule)
    pl.LightningDataModule.__init__(datamodule)
    datamodule.task = task
    datamodule.epoch = 0
    datamodule.config = DictConfig(
        {
            "dataloader": {
                "batch_size": {"training": batch_size, "validation": batch_size, "test": batch_size},
                "num_workers": {"training": num_workers, "validation": num_workers, "test": num_workers},
                "pin_memory": False,
                "prefetch_factor": 1,
                "persistent_workers": persistent_workers,
                "loading_policy": {"_target_": "anemoi.training.data.loading.SynchronizedLoadingPolicy"},
            },
        },
    )
    return datamodule


def _flatten(batches: object) -> list[int]:
    """Flatten the batches a BatchSampler yields into a flat list of sample indices."""
    return [index for batch in batches for index in batch]


def _attach_statistics_reader(
    datamodule: AnemoiDatasetsDataModule,
    mocker: MockFixture,
) -> Mock:
    reader = mocker.Mock()
    reader.statistics_tendencies.side_effect = lambda delta: {"delta": delta}
    datamodule.__dict__["ds_train"] = SimpleNamespace(data_readers={"data": reader})
    return reader


def test_forecaster_uses_cumulative_tendency_statistics_for_each_output_step(mocker: MockFixture) -> None:
    """Reference-to-lead targets use statistics for their cumulative lead times."""
    task = Forecaster(multistep_input=1, multistep_output=3, timestep="6h")
    datamodule = _make_datamodule(task)
    reader = _attach_statistics_reader(datamodule, mocker)

    statistics = datamodule.statistics_tendencies

    assert statistics == {
        "data": {
            "6h": {"delta": "6h"},
            "12h": {"delta": "12h"},
            "18h": {"delta": "18h"},
            "lead_times": ["6h", "12h", "18h"],
        },
    }
    assert [call.args[0] for call in reader.statistics_tendencies.call_args_list] == ["6h", "12h", "18h"]


def test_mixed_frequency_tendency_statistics_use_dataset_output_times(mocker: MockFixture) -> None:
    task = Forecaster(multistep_input=1, multistep_output=12, timestep="5m")
    task.model_timestep = frequency_to_timedelta("5m")
    task.dataset_target_relative_times_by_dataset_by_step = {"data": [[12]]}
    datamodule = _make_datamodule(task)
    reader = _attach_statistics_reader(datamodule, mocker)

    statistics = datamodule.statistics_tendencies

    assert statistics == {
        "data": {
            "1h": {"delta": "1h"},
            "lead_times": ["1h"],
        },
    }
    reader.statistics_tendencies.assert_called_once_with("1h")


def test_temporal_downscaler_uses_cumulative_tendency_statistics_per_lead_time(mocker: MockFixture) -> None:
    """TemporalDownscaler uses per-lead-time cumulative tendency statistics."""
    task = TemporalDownscaler(input_timestep="6h", output_timestep="2h")
    datamodule = _make_datamodule(task)
    reader = _attach_statistics_reader(datamodule, mocker)

    statistics = datamodule.statistics_tendencies

    assert statistics == {
        "data": {
            "2h": {"delta": "2h"},
            "4h": {"delta": "4h"},
            "lead_times": ["2h", "4h"],
        },
    }
    assert [call.args[0] for call in reader.statistics_tendencies.call_args_list] == ["2h", "4h"]


@pytest.mark.parametrize(
    "rollout",
    [
        pytest.param({"start": 1, "epoch_increment": 0, "maximum": 1}, id="fixed"),
        pytest.param({"start": 1, "epoch_increment": 1, "maximum": 3}, id="progressing"),
    ],
)
@pytest.mark.parametrize("persistent_workers", [False, True])
def test_persistent_workers_follow_dataloader_config(rollout: dict[str, int], persistent_workers: bool) -> None:
    """All dataloaders use the configured persistence, whatever the rollout schedule."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h", rollout=rollout)
    datamodule = _make_datamodule(task, persistent_workers=persistent_workers)

    loaders = [datamodule._get_dataloader(SampleIndexDataset(), stage) for stage in ("training", "validation", "test")]

    assert [loader.persistent_workers for loader in loaders] == [persistent_workers] * len(loaders)


def test_persistent_workers_default_to_false_when_config_is_unvalidated() -> None:
    """The documented default applies when validation does not populate the field."""
    datamodule = _make_datamodule(_fixed_rollout_task(), persistent_workers=True)
    del datamodule.config.dataloader.persistent_workers

    loader = datamodule._get_dataloader(SampleIndexDataset(), "training")

    assert loader.persistent_workers is False


def test_set_epoch_updates_all_constructed_datasets(mocker: MockFixture) -> None:
    """set_epoch updates every already-cached dataset and leaves lazy datasets untouched."""
    datamodule = AnemoiDatasetsDataModule.__new__(AnemoiDatasetsDataModule)
    datamodule.epoch = 0
    datamodule.task = mocker.Mock()

    ds_train = mocker.Mock()
    ds_train.data_readers = {"data": object()}
    ds_valid = mocker.Mock()
    ds_valid.data_readers = {"data": object()}
    ds_test = mocker.Mock()
    ds_test.data_readers = {"data": object()}
    datamodule.__dict__.update(ds_train=ds_train, ds_valid=ds_valid, ds_test=ds_test)
    datamodule.__dict__["loading_policy"] = mocker.Mock()
    datamodule.loading_policy.relative_date_indices.side_effect = lambda _task, _data_readers, mode: {"data": [mode]}

    datamodule.set_epoch(5)

    assert datamodule.epoch == 5
    ds_train.set_relative_date_indices.assert_called_once_with(
        relative_date_indices={"data": ["training"]},
    )
    ds_valid.set_relative_date_indices.assert_called_once_with(
        relative_date_indices={"data": ["validation"]},
    )
    ds_test.set_relative_date_indices.assert_called_once_with(
        relative_date_indices={"data": ["test"]},
    )


def test_get_dataset_loads_time_steps_of_current_rollout(mocker: MockFixture) -> None:
    """Datasets load the time steps that the task needs when they are constructed."""
    datamodule = AnemoiDatasetsDataModule.__new__(AnemoiDatasetsDataModule)
    datamodule.task = mocker.Mock()
    datamodule.__dict__["loading_policy"] = mocker.Mock()

    data_reader = mocker.Mock()
    data_reader.frequency = "6h"
    create_dataset = mocker.patch("anemoi.training.data.datamodule.create_dataset", return_value=data_reader)
    datamodule.loading_policy.relative_date_indices.return_value = {"data": [0, 1]}

    datamodule._get_dataset({"data": object()}, label="validation")

    create_dataset.assert_called_once()
    datamodule.loading_policy.build_dataset.assert_called_once_with(
        {"data": data_reader},
        {"data": [0, 1]},
    )


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
        self.num_sequences = 1
        self.missing_sequences: set[int] = set()
        self.default_sampling = {"stride": 1}
        self.statistics = {}
        self.metadata = {}
        self.supporting_arrays = {}
        self.variables = ["forcing_var", "prog_var"]
        self.name_to_index = {"forcing_var": 0, "prog_var": 1}
        self.resolution = "test"

    def sequence_length(self, _sequence: int = 0) -> int:
        return len(self.dates)

    def missing_positions(self, _sequence: int = 0) -> set[int]:
        return self.missing

    def compute_anchors(self, relative_indices: list[int] | np.ndarray, _sampling: dict | None = None) -> np.ndarray:
        from anemoi.training.data.usable_indices import get_usable_indices

        rel = np.asarray(list(relative_indices), dtype=np.int64)
        positions = get_usable_indices(self.missing, len(self.dates), rel)
        seq_col = np.zeros(positions.size, dtype=np.int64)
        return np.stack([seq_col, positions], axis=1)

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
                "loading_policy": {
                    "_target_": "anemoi.training.data.loading.MixedFrequencyLoadingPolicy",
                    "frequency": "5m",
                },
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
                "loading_policy": {
                    "_target_": "anemoi.training.data.loading.MixedFrequencyLoadingPolicy",
                    "frequency": "5m",
                },
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

    assert datamodule.ds_train.model_relative_date_indices.tolist() == [0, 1]


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
                "loading_policy": {
                    "_target_": "anemoi.training.data.loading.MixedFrequencyLoadingPolicy",
                    "frequency": "6h",
                },
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

    assert datamodule.ds_train.model_relative_date_indices.tolist() == [-1, 0, 1]


def test_datamodule_keeps_dense_path_for_aligned_frequencies(
    mocker: MockFixture,
) -> None:
    cfg = make_multidataset_cfg(meps_frequency="5m", radar_frequency="5m")
    cfg.dataloader.loading_policy = {
        "_target_": "anemoi.training.data.loading.SynchronizedLoadingPolicy",
    }
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
    assert task.num_input_timesteps_by_dataset == {"meps": 1, "nordic_radar": 1}
    assert task.num_output_timesteps_by_dataset == {"meps": 0, "nordic_radar": 1}


def test_mixed_frequency_indices_remain_anchored_at_forecast_initialization(
    mocker: MockFixture,
) -> None:
    cfg = make_multidataset_cfg(meps_frequency="6h", radar_frequency="5m")
    task = Forecaster(
        multistep_input=12,
        multistep_output=1,
        timestep="5m",
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

    assert datamodule.ds_train.model_relative_date_indices.tolist() == list(range(-11, 2))
    assert datamodule.ds_train.model_relative_date_indices_by_dataset["meps"].tolist() == [0]


def test_mixed_frequency_rollout_window_updates_with_epoch(mocker: MockFixture) -> None:
    cfg = make_multidataset_cfg(meps_frequency="1h", radar_frequency="5m")
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="5m",
        rollout={"start": 1, "epoch_increment": 1, "maximum": 2},
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
    dataset = datamodule.ds_train

    task.on_train_epoch_end(0)
    datamodule.set_epoch(1)

    assert dataset.model_relative_date_indices.tolist() == [0, 1, 2]
    assert task.dataset_target_relative_times_by_dataset_by_step == {
        "meps": [[], []],
        "nordic_radar": [[1], [2]],
    }


def test_mixed_frequency_input_only_dataset_remains_target_free_at_native_cadence(
    mocker: MockFixture,
) -> None:
    cfg = make_multidataset_cfg(meps_frequency="6h", radar_frequency="5m")
    cfg.model = {
        "decoders": {
            "radar": {"target_datasets": ["nordic_radar"]},
        },
    }
    task = Forecaster(
        multistep_input=1,
        multistep_output=12,
        timestep="5m",
        rollout={"start": 6, "epoch_increment": 0, "maximum": 6},
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
    metadata = {"metadata_inference": {}}

    datamodule.fill_metadata(metadata)
    task.fill_metadata(metadata)
    task.configure_from_metadata(metadata)

    assert datamodule.ds_train.model_relative_date_indices_by_dataset["meps"].tolist() == [0, 72]
    assert task.dataset_target_relative_times_by_dataset_by_step["meps"] == [[]] * 6
    assert task.dataset_time_maps["meps"] == {0: 0, 72: 1}
    assert task.num_output_timesteps_by_dataset["meps"] == 0
    assert task.num_output_timesteps_by_dataset["nordic_radar"] == 12


def test_offset_forecaster_uses_signed_mixed_frequency_offsets(mocker: MockFixture) -> None:
    cfg = make_multidataset_cfg(meps_frequency="12h", radar_frequency="6h", data_frequency="6h")
    del cfg.task.timestep
    cfg.dataloader.loading_policy.frequency = "6h"
    task = OffsetForecaster(
        input_offsets=["-12h", "0h"],
        output_offsets=["6h", "12h"],
        rollout_shift="12h",
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
    metadata = {"metadata_inference": {}}

    datamodule.fill_metadata(metadata)
    task.fill_metadata(metadata)
    task.configure_from_metadata(metadata)

    assert datamodule.ds_train.model_relative_date_indices.tolist() == [-2, 0, 1, 2]
    assert task.dataset_input_relative_times_by_dataset == {
        "meps": [-2, 0],
        "nordic_radar": [-2, 0],
    }
    assert task.dataset_target_relative_times_by_dataset_by_step == {
        "meps": [[2]],
        "nordic_radar": [[1, 2]],
    }


def test_state_dict_restores_dataloader_epoch() -> None:
    """Checkpoint state restores the epoch used by datasets and new workers."""
    datamodule = AnemoiDatasetsDataModule.__new__(AnemoiDatasetsDataModule)
    datamodule.epoch = 4

    state = datamodule.state_dict()

    resumed_datamodule = AnemoiDatasetsDataModule.__new__(AnemoiDatasetsDataModule)
    resumed_datamodule.epoch = 0
    resumed_datamodule.load_state_dict(state)

    assert state == {"epoch": 4}
    assert resumed_datamodule.epoch == 4


@pytest.mark.parametrize(
    ("rollout", "expected"),
    [
        pytest.param({"start": 1, "epoch_increment": 0, "maximum": 1}, False, id="fixed"),
        pytest.param({"start": 1, "epoch_increment": 1, "maximum": 3}, True, id="progressing"),
    ],
)
def test_rollout_changes_between_epochs(rollout: dict[str, int], expected: bool) -> None:
    """Dataloaders are rebuilt every epoch only when the rollout schedule can change the samples."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h", rollout=rollout)

    assert _make_datamodule(task).rollout_changes_between_epochs is expected


def test_training_batch_sampler_shuffles_by_base_seed_and_epoch() -> None:
    """The training order is the same for the same base seed and epoch and changes between epochs."""
    datamodule = _make_datamodule(_fixed_rollout_task())
    dataset = SampleIndexDataset(40)

    batch_sampler = datamodule._get_batch_sampler(dataset, "training")
    assert batch_sampler.sampler.seed == derive_seed(1000, SeedContext.DATALOADER)
    order_epoch_0 = _flatten(batch_sampler)

    datamodule.epoch = 3
    order_epoch_3 = _flatten(datamodule._get_batch_sampler(dataset, "training"))
    repeated_order_epoch_3 = _flatten(datamodule._get_batch_sampler(dataset, "training"))

    assert sorted(order_epoch_0) == list(range(40))
    assert order_epoch_0 != list(range(40))
    assert order_epoch_3 != order_epoch_0
    assert repeated_order_epoch_3 == order_epoch_3


def test_validation_batch_sampler_keeps_order() -> None:
    """Validation and test samples are not shuffled."""
    datamodule = _make_datamodule(_fixed_rollout_task())

    for stage in ("validation", "test"):
        assert _flatten(datamodule._get_batch_sampler(SampleIndexDataset(6), stage)) == list(range(6))


def test_batch_sampler_splits_samples_between_sample_groups(mocker: MockFixture) -> None:
    """Each sample group gets an equal, separate share of the samples."""
    datamodule = _make_datamodule(_fixed_rollout_task())
    datamodule.trainer = mocker.Mock()
    dataset = SampleIndexDataset(11)

    shares = []
    for rank in range(3):
        datamodule.trainer.distributed_sampler_kwargs = {"num_replicas": 3, "rank": rank}
        shares.append(_flatten(datamodule._get_batch_sampler(dataset, "training")))

    assert [len(share) for share in shares] == [3, 3, 3]
    assert len(set().union(*shares)) == 9


def test_batch_sampler_uses_single_group_without_distributed_strategy(mocker: MockFixture) -> None:
    """Without a distributed strategy one rank loads all samples."""
    datamodule = _make_datamodule(_fixed_rollout_task())
    datamodule.trainer = mocker.Mock(distributed_sampler_kwargs=None)

    assert sorted(_flatten(datamodule._get_batch_sampler(SampleIndexDataset(5), "training"))) == list(range(5))


class _InterruptError(Exception):
    """Stops a training run after its checkpoint was saved."""


class _StopAfterCheckpoint(pl.Callback):
    """Save a checkpoint at the given point of training and stop."""

    def __init__(self, path: Path, *, global_step: int | None = None, validation_end_of_epoch: int | None = None):
        self.path = path
        self.global_step = global_step
        self.validation_end_of_epoch = validation_end_of_epoch

    def on_train_batch_end(self, trainer: pl.Trainer, *args, **kwargs) -> None:
        del args, kwargs
        if trainer.global_step == self.global_step:
            trainer.save_checkpoint(self.path, weights_only=False)
            raise _InterruptError

    def on_validation_end(self, trainer: pl.Trainer, *args, **kwargs) -> None:
        del args, kwargs
        if not trainer.sanity_checking and trainer.current_epoch == self.validation_end_of_epoch:
            trainer.save_checkpoint(self.path, weights_only=False)
            raise _InterruptError


class _RecordingModule(pl.LightningModule):
    """Record the samples of every training batch."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        self.seen: list[tuple[int, list[int]]] = []

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        del batch_idx
        self.seen.append((self.current_epoch, batch.tolist()))
        return self.layer(batch.float().unsqueeze(-1)).sum()

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        del batch, batch_idx

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.0)

    def on_train_epoch_end(self) -> None:
        self.trainer.datamodule.set_epoch(self.current_epoch + 1)


def _fit(
    mocker: MockFixture,
    *,
    callbacks: list[pl.Callback] | None = None,
    ckpt_path: Path | None = None,
) -> list[tuple[int, list[int]]]:
    """Train two epochs as the second of two sample groups and return the samples of every batch."""
    datamodule = _make_datamodule(_fixed_rollout_task(), batch_size=3, num_workers=2, persistent_workers=True)
    datamodule.__dict__.update(ds_train=SampleIndexDataset(40), ds_valid=SampleIndexDataset(6))
    mocker.patch.object(datamodule, "_sample_groups", return_value={"num_replicas": 2, "rank": 1})
    module = _RecordingModule()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        use_distributed_sampler=False,
        callbacks=callbacks or [],
    )
    with contextlib.suppress(_InterruptError):
        trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)
    return module.seen


@pytest.mark.parametrize(
    "stop",
    [
        pytest.param({"global_step": 3}, id="mid-first-epoch"),
        pytest.param({"global_step": 10}, id="mid-second-epoch"),
        pytest.param({"validation_end_of_epoch": 0}, id="end-of-first-epoch"),
    ],
)
def test_resumed_training_loads_the_same_batches(
    mocker: MockFixture,
    tmp_path: Path,
    stop: dict[str, int],
) -> None:
    """Training resumed from a checkpoint continues with the batches an uninterrupted run loads."""
    checkpoint = tmp_path / "interrupted.ckpt"

    uninterrupted = _fit(mocker)
    interrupted = _fit(mocker, callbacks=[_StopAfterCheckpoint(checkpoint, **stop)])
    resumed = _fit(mocker, ckpt_path=checkpoint)

    # 20 samples per sample group in batches of 3 give 7 batches per epoch.
    assert len(uninterrupted) == 14
    assert interrupted + resumed == uninterrupted
