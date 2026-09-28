# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import contextlib
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytest_mock import MockFixture
from torch.utils.data import Dataset

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule
from anemoi.training.tasks import Forecaster
from anemoi.training.tasks import TemporalDownscaler
from anemoi.training.tasks.base import BaseTask
from anemoi.training.utils.seeding import SeedContext
from anemoi.training.utils.seeding import derive_seed


class SampleIndexDataset(Dataset):
    """Returns the index of each sample, so batches show which samples were loaded."""

    def __init__(self, size: int = 1) -> None:
        self.size = size
        self.data_readers: dict = {}

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


def _make_datamodule(
    task: BaseTask,
    *,
    persistent_workers: bool = True,
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

    loaders = [datamodule._get_dataloader(SampleIndexDataset(), stage) for stage in ("training", "validation", "test")]

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

    loader = datamodule._get_dataloader(SampleIndexDataset(), "training")

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

    loaders = [datamodule._get_dataloader(SampleIndexDataset(), stage) for stage in ("training", "validation", "test")]

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
    ds_train.set_relative_date_indices.assert_called_once_with({"data": ["training"]})
    ds_valid.set_relative_date_indices.assert_called_once_with({"data": ["validation"]})
    ds_test.set_relative_date_indices.assert_called_once_with({"data": ["test"]})


def test_get_dataset_loads_time_steps_of_current_rollout(mocker: MockFixture) -> None:
    """Datasets load the time steps that the task needs when they are constructed."""
    datamodule = AnemoiDatasetsDataModule.__new__(AnemoiDatasetsDataModule)
    datamodule.task = mocker.Mock()

    data_reader = object()
    create_dataset = mocker.patch("anemoi.training.data.datamodule.create_dataset", return_value=data_reader)
    mocker.patch(
        "anemoi.training.data.datamodule.compute_relative_date_indices",
        return_value={"data": [0, 1]},
    )
    multi_dataset = mocker.patch("anemoi.training.data.datamodule.MultiDataset")

    datamodule._get_dataset({"data": object()}, label="validation")

    create_dataset.assert_called_once()
    multi_dataset.assert_called_once_with(
        data_readers={"data": data_reader},
        relative_date_indices={"data": [0, 1]},
    )


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


def test_training_sampler_shuffles_by_base_seed_and_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The training order is the same for the same base seed and epoch and changes between epochs."""
    monkeypatch.setenv("ANEMOI_BASE_SEED", "1000")
    datamodule = _make_datamodule(_fixed_rollout_task())
    dataset = SampleIndexDataset(40)

    sampler = datamodule._get_sampler(dataset, "training")
    assert sampler.seed == derive_seed(1000, SeedContext.DATALOADER)
    order_epoch_0 = list(sampler)

    datamodule.epoch = 3
    order_epoch_3 = list(datamodule._get_sampler(dataset, "training"))
    repeated_order_epoch_3 = list(datamodule._get_sampler(dataset, "training"))

    assert sorted(order_epoch_0) == list(range(40))
    assert order_epoch_0 != list(range(40))
    assert order_epoch_3 != order_epoch_0
    assert repeated_order_epoch_3 == order_epoch_3


def test_validation_sampler_keeps_order() -> None:
    """Validation and test samples are not shuffled."""
    datamodule = _make_datamodule(_fixed_rollout_task())

    for stage in ("validation", "test"):
        assert list(datamodule._get_sampler(SampleIndexDataset(6), stage)) == list(range(6))


def test_sampler_splits_samples_between_sample_groups(mocker: MockFixture) -> None:
    """Each sample group gets an equal, separate share of the samples."""
    datamodule = _make_datamodule(_fixed_rollout_task())
    datamodule.trainer = mocker.Mock()
    dataset = SampleIndexDataset(11)

    shares = []
    for rank in range(3):
        datamodule.trainer.distributed_sampler_kwargs = {"num_replicas": 3, "rank": rank}
        shares.append(list(datamodule._get_sampler(dataset, "training")))

    assert [len(share) for share in shares] == [3, 3, 3]
    assert len(set().union(*shares)) == 9


def test_sampler_uses_single_group_without_distributed_strategy(mocker: MockFixture) -> None:
    """Without a distributed strategy one rank loads all samples."""
    datamodule = _make_datamodule(_fixed_rollout_task())
    datamodule.trainer = mocker.Mock(distributed_sampler_kwargs=None)

    assert sorted(datamodule._get_sampler(SampleIndexDataset(5), "training")) == list(range(5))


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
    mocker.patch("anemoi.training.data.datamodule.compute_relative_date_indices", return_value={})
    datamodule = _make_datamodule(_fixed_rollout_task(), batch_size=3, num_workers=2, persistent_workers=True)
    datamodule.__dict__.update(ds_train=SampleIndexDataset(40), ds_valid=SampleIndexDataset(6))
    mocker.patch.object(datamodule, "_sampler_group", return_value={"num_replicas": 2, "rank": 1})
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
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stop: dict[str, int],
) -> None:
    """Training resumed from a checkpoint continues with the batches an uninterrupted run loads."""
    monkeypatch.setenv("ANEMOI_BASE_SEED", "1000")
    checkpoint = tmp_path / "interrupted.ckpt"

    uninterrupted = _fit(mocker)
    interrupted = _fit(mocker, callbacks=[_StopAfterCheckpoint(checkpoint, **stop)])
    resumed = _fit(mocker, ckpt_path=checkpoint)

    # 20 samples per sample group in batches of 3 give 7 batches per epoch.
    assert len(uninterrupted) == 14
    assert interrupted + resumed == uninterrupted
