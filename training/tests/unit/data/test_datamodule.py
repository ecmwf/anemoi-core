# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytest_mock import MockFixture
from torch.utils.data import IterableDataset

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule
from anemoi.training.tasks import Forecaster
from anemoi.training.tasks import OffsetForecaster
from anemoi.training.tasks import TemporalDownscaler
from anemoi.training.tasks.base import BaseTask
from anemoi.utils.dates import frequency_to_seconds
from anemoi.utils.dates import frequency_to_timedelta


class TinyIterableDataset(IterableDataset):
    """Minimal iterable dataset for DataLoader construction tests."""

    def __iter__(self) -> Iterator[int]:
        yield 0


def _make_datamodule(task: BaseTask, *, persistent_workers: bool = True) -> AnemoiDatasetsDataModule:
    datamodule = AnemoiDatasetsDataModule.__new__(AnemoiDatasetsDataModule)
    datamodule.task = task
    datamodule.config = DictConfig(
        {
            "dataloader": {
                "batch_size": {"training": 1, "validation": 1, "test": 1},
                "num_workers": {"training": 1, "validation": 1, "test": 1},
                "pin_memory": False,
                "prefetch_factor": 1,
                "persistent_workers": persistent_workers,
            },
        },
    )
    return datamodule


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


@pytest.mark.parametrize("persistent_workers", [False, True])
def test_persistent_workers_follow_dataloader_config(persistent_workers: bool) -> None:
    """All dataloaders use the configured persistence when rollout is fixed."""
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 1, "epoch_increment": 0, "maximum": 1},
    )
    datamodule = _make_datamodule(task, persistent_workers=persistent_workers)

    loaders = [datamodule._get_dataloader(TinyIterableDataset(), stage) for stage in ("training", "validation", "test")]

    assert [loader.persistent_workers for loader in loaders] == [persistent_workers] * len(loaders)


def test_persistent_workers_default_to_true_when_config_is_unvalidated() -> None:
    """The documented default applies when validation does not populate the field."""
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 1, "epoch_increment": 0, "maximum": 1},
    )
    datamodule = _make_datamodule(task)
    del datamodule.config.dataloader.persistent_workers

    loader = datamodule._get_dataloader(TinyIterableDataset(), "training")

    assert loader.persistent_workers is True


@pytest.mark.parametrize(
    "rollout",
    [
        pytest.param({"start": 1, "epoch_increment": 1, "maximum": 3}, id="progressing"),
        pytest.param({"start": 3, "epoch_increment": 1, "maximum": 3}, id="at-maximum"),
    ],
)
def test_persistent_workers_are_disabled_for_rollout_schedule(
    rollout: dict[str, int],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An epoch increment disables persistence even after rollout reaches its maximum."""
    caplog.set_level(logging.INFO)
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout=rollout,
    )
    datamodule = _make_datamodule(task, persistent_workers=True)

    loaders = [datamodule._get_dataloader(TinyIterableDataset(), stage) for stage in ("training", "validation", "test")]

    assert [loader.persistent_workers for loader in loaders] == [False] * len(loaders)
    assert datamodule.config.dataloader.persistent_workers is True
    assert caplog.messages == [
        "Disabling dataloader.persistent_workers because the rollout changes between epochs.",
    ]


def test_set_epoch_updates_all_constructed_datasets(mocker: MockFixture) -> None:
    """set_epoch updates every already-cached dataset and leaves lazy datasets untouched."""
    datamodule = AnemoiDatasetsDataModule.__new__(AnemoiDatasetsDataModule)
    datamodule.epoch = 0
    datamodule.task = mocker.Mock()
    datamodule.task.steps.side_effect = lambda label: tuple(
        {} for _ in range({"training": 1, "validation": 2, "test": 3}[label])
    )

    ds_train = mocker.Mock()
    ds_train.data_readers = {"data": object()}
    ds_valid = mocker.Mock()
    ds_valid.data_readers = {"data": object()}
    ds_test = mocker.Mock()
    ds_test.data_readers = {"data": object()}
    datamodule.__dict__.update(ds_train=ds_train, ds_valid=ds_valid, ds_test=ds_test)

    mocker.patch(
        "anemoi.training.data.datamodule.compute_relative_date_indices",
        side_effect=lambda _task, _data_readers, mode: {"data": [mode]},
    )

    datamodule.set_epoch(5)

    assert datamodule.epoch == 5
    ds_train.set_epoch.assert_called_once_with(
        5,
        rollout=1,
        relative_date_indices={"data": ["training"]},
    )
    ds_valid.set_epoch.assert_called_once_with(
        5,
        rollout=2,
        relative_date_indices={"data": ["validation"]},
    )
    ds_test.set_epoch.assert_called_once_with(
        5,
        rollout=3,
        relative_date_indices={"data": ["test"]},
    )


def test_get_dataset_uses_current_epoch_for_lazy_construction(mocker: MockFixture) -> None:
    """Datasets constructed after set_epoch receive the datamodule's current epoch."""
    datamodule = AnemoiDatasetsDataModule.__new__(AnemoiDatasetsDataModule)
    datamodule.epoch = 7
    datamodule.task = mocker.Mock()
    datamodule.task.steps.return_value = ({}, {})

    data_reader = mocker.Mock()
    data_reader.frequency = "6h"
    create_dataset = mocker.patch("anemoi.training.data.datamodule.create_dataset", return_value=data_reader)
    mocker.patch(
        "anemoi.training.data.datamodule.compute_relative_date_indices",
        return_value={"data": [0, 1]},
    )
    multi_dataset = mocker.patch("anemoi.training.data.datamodule.MultiDataset")

    datamodule._get_dataset({"data": object()}, shuffle=False, label="validation")

    create_dataset.assert_called_once()
    multi_dataset.assert_called_once_with(
        data_readers={"data": data_reader},
        relative_date_indices={"data": [0, 1]},
        shuffle=False,
        label="validation",
        epoch=7,
        rollout=2,
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
    dataset.per_worker_init(n_workers=1, worker_id=0)

    task.on_train_epoch_end(0)
    datamodule.set_epoch(1)

    assert dataset.epoch == 1
    assert dataset.rollout == 2
    assert dataset.model_relative_date_indices.tolist() == [0, 1, 2]
    assert dataset.chunk_index_range is None
    assert task.dataset_target_relative_times_by_dataset_by_step == {
        "meps": [[], []],
        "nordic_radar": [[1], [2]],
    }


def test_offset_forecaster_uses_signed_mixed_frequency_offsets(mocker: MockFixture) -> None:
    cfg = make_multidataset_cfg(meps_frequency="12h", radar_frequency="6h", data_frequency="6h")
    del cfg.task.timestep
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
