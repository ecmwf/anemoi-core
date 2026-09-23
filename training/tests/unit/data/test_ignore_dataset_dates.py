# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from anemoi.training.data.data_reader import NativeGridDataset
from anemoi.training.data.data_reader import create_dataset
from anemoi.training.data.datamodule import AnemoiDatasetsDataModule
from anemoi.training.schemas.dataloader import DataLoaderSchema
from anemoi.training.tasks import Forecaster


class BrokenDatesDataset:
    shape = (6, 2, 1, 3)
    frequency = datetime.timedelta(hours=6)
    missing = frozenset()

    @property
    def dates(self) -> np.ndarray:
        msg = "Synthetic dates array is shorter than the data"
        raise IndexError(msg)

    def __getitem__(self, index: tuple) -> np.ndarray:
        return np.arange(36, dtype=np.float32).reshape(self.shape)[index]


@pytest.mark.parametrize(("start", "end"), [(None, 2020), (2021, 2021), (2022, None)])
@pytest.mark.parametrize("frequency", [None, "6h", "360m"])
def test_index_only_reader_ignores_dates_and_preserves_data(
    start: int | None,
    end: int | None,
    frequency: str | None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = {"dataset": "synthetic.zarr", "frequency": frequency, "drop": []}
    config = {"dataset_config": source, "start": start, "end": end}
    original = deepcopy(config)
    open_dataset = Mock(return_value=BrokenDatesDataset())
    monkeypatch.setattr("anemoi.training.data.data_reader.open_dataset", open_dataset)

    reader = create_dataset(config, ignore_dates=True)

    open_dataset.assert_called_once_with({"dataset": "synthetic.zarr", "drop": []})
    assert config == original
    assert reader.frequency == datetime.timedelta(hours=6)
    assert reader.sequence_length() == 6
    np.testing.assert_array_equal(reader.compute_anchors([-1, 0, 1]), [[0, 1], [0, 2], [0, 3], [0, 4]])
    sample = reader.get_sample(0, slice(4, 6), slice(1, 3))
    expected = np.arange(36, dtype=np.float32).reshape(6, 2, 1, 3)[4:6, :, :, 1:3].transpose(0, 2, 3, 1)
    torch.testing.assert_close(sample, torch.from_numpy(expected))
    assert "periods may overlap" in caplog.text


def test_normal_reader_still_uses_date_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    data = Mock()
    data.dates = np.arange(5)
    data.shape = (6, 2, 1, 3)
    open_dataset = Mock(return_value=data)
    monkeypatch.setattr("anemoi.training.data.data_reader.open_dataset", open_dataset)
    source = {"dataset": "normal.zarr", "frequency": "12h"}

    reader = NativeGridDataset(dataset_config=source, start=2019, end=2020)

    open_dataset.assert_called_once_with(source, start=2019, end=2020)
    assert reader.sequence_length() == 5
    assert not reader.ignore_dates


def test_index_only_reader_rejects_resampling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("anemoi.training.data.data_reader.open_dataset", Mock(return_value=BrokenDatesDataset()))
    with pytest.raises(ValueError, match="native frequency"):
        NativeGridDataset(dataset_config={"dataset": "synthetic.zarr", "frequency": "12h"}, ignore_dates=True)


@pytest.mark.parametrize(
    "config",
    [
        {"dataset_config": "trajectory.zarr", "trajectory": {}},
        {"dataset_config": {"dataset": {"join": ["a.zarr", "b.zarr"]}}},
    ],
)
def test_index_only_reader_rejects_unsupported_layouts(config: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    open_dataset = Mock()
    monkeypatch.setattr("anemoi.training.data.data_reader.open_dataset", open_dataset)
    with pytest.raises(ValueError, match="ignore_dataset_dates"):
        create_dataset(config, ignore_dates=True)
    open_dataset.assert_not_called()


@pytest.mark.parametrize("stage", ["training", "validation", "test"])
def test_datamodule_enables_index_sampling_for_each_stage(
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    data = BrokenDatesDataset()
    monkeypatch.setattr("anemoi.training.data.data_reader.open_dataset", Mock(return_value=data))
    split = {"datasets": {"data": {"dataset_config": "synthetic.zarr", "start": None, "end": None}}}
    config = DictConfig(
        {
            "dataloader": {
                "ignore_dataset_dates": True,
                "pin_memory": False,
                "training": split,
                "validation": split,
                "test": split,
            },
        },
    )
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")
    datamodule = AnemoiDatasetsDataModule(config, task)
    dataset = getattr(datamodule, {"training": "ds_train", "validation": "ds_valid", "test": "ds_test"}[stage])

    assert dataset.data_readers["data"].ignore_dates
    assert dataset.data_readers["data"].sequence_length() == 6
    assert len(dataset.anchors) == 5
    assert "periods may overlap" in caplog.text


def test_datamodule_still_requires_training_end_date() -> None:
    split = {"datasets": {"data": {"dataset_config": "normal.zarr", "start": None, "end": None}}}
    config = DictConfig({"dataloader": {"training": split, "validation": split, "test": split}})
    with pytest.raises(ValueError, match="No end date specified"):
        AnemoiDatasetsDataModule(config, Forecaster(multistep_input=1, multistep_output=1, timestep="6h"))


def test_date_bypass_is_opt_in_in_schema() -> None:
    assert DataLoaderSchema.model_fields["ignore_dataset_dates"].default is False
