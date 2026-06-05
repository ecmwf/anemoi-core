# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Iterator

import pytest
from omegaconf import DictConfig
from pytest_mock import MockFixture
from torch.utils.data import IterableDataset

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule
from anemoi.training.tasks import Forecaster


class TinyIterableDataset(IterableDataset):
    """Minimal iterable dataset for DataLoader construction tests."""

    def __iter__(self) -> Iterator[int]:
        yield 0


def _make_datamodule(task: Forecaster) -> AnemoiDatasetsDataModule:
    datamodule = AnemoiDatasetsDataModule.__new__(AnemoiDatasetsDataModule)
    datamodule.task = task
    datamodule.config = DictConfig(
        {
            "dataloader": {
                "batch_size": {"training": 1, "validation": 1, "test": 1},
                "num_workers": {"training": 1, "validation": 1, "test": 1},
                "pin_memory": False,
                "prefetch_factor": 1,
            },
        },
    )
    return datamodule


@pytest.mark.parametrize(
    ("rollout", "persistent_workers"),
    [
        ({"start": 1, "epoch_increment": 1, "maximum": 3}, False),
        ({"start": 1, "epoch_increment": 0, "maximum": 3}, True),
        ({"start": 3, "epoch_increment": 1, "maximum": 3}, False),
    ],
)
def test_persistent_workers_follow_shared_rollout_progression_policy(
    rollout: dict[str, int],
    persistent_workers: bool,
) -> None:
    """All dataloaders share the same persistence policy based on rollout progression."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h", rollout=rollout)
    datamodule = _make_datamodule(task)

    loaders = [datamodule._get_dataloader(TinyIterableDataset(), stage) for stage in ("training", "validation", "test")]

    assert [loader.persistent_workers for loader in loaders] == [persistent_workers] * len(loaders)


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

    data_reader = object()
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
