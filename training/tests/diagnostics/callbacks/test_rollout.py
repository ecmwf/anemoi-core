# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

import pytest

from anemoi.training.diagnostics.callbacks.rollout import UpdateRollout
from anemoi.training.schedulers.rollout import RolloutScheduler
from anemoi.training.train.forecaster import GraphForecaster
from anemoi.training.train.train import AnemoiTrainer


class DebugScheduler(RolloutScheduler):
    @property
    def rollout(self) -> int:
        return self._epoch

    @property
    def maximum_rollout(self) -> int:
        return self._epoch

    def description(self) -> str:
        return "DebugScheduler"


@pytest.fixture
def fake_trainer(mocker: Any) -> AnemoiTrainer:
    trainer = mocker.Mock(spec=AnemoiTrainer)

    trainer.datamodule.update_rollout = mocker.patch(
        "anemoi.training.data.datamodule.AnemoiDatasetsDataModule.update_rollout",
    )
    return trainer


@pytest.fixture
def fake_forecaster(mocker: Any) -> GraphForecaster:
    model = mocker.Mock(spec=GraphForecaster)

    model.rollout = DebugScheduler()
    return model


@pytest.fixture
def checkpoint() -> dict[str, int]:
    return {"epoch": 10, "global_step": 100}


@pytest.fixture
def callback() -> UpdateRollout:
    callback = UpdateRollout()
    assert callback is not None
    assert hasattr(callback, "on_load_checkpoint")
    assert hasattr(callback, "on_validation_epoch_end")

    return callback


def test_on_load_checkpoint(
    fake_trainer: AnemoiTrainer,
    fake_forecaster: GraphForecaster,
    callback: UpdateRollout,
    checkpoint: dict,
) -> None:
    callback.on_load_checkpoint(fake_trainer, fake_forecaster, checkpoint)
    spy = fake_trainer.datamodule.update_rollout

    spy.assert_called_once_with(rollout=checkpoint["epoch"])


def test_on_validation_epoch_sanity(
    fake_trainer: AnemoiTrainer,
    fake_forecaster: GraphForecaster,
    callback: UpdateRollout,
) -> None:
    fake_trainer.current_epoch = 10
    fake_trainer.sanity_checking = True
    spy = fake_trainer.datamodule.update_rollout

    callback.on_validation_epoch_end(fake_trainer, fake_forecaster, None)
    spy.assert_not_called()


def test_on_validation_epoch(
    fake_trainer: AnemoiTrainer,
    fake_forecaster: GraphForecaster,
    callback: UpdateRollout,
) -> None:
    fake_trainer.current_epoch = 10
    spy = fake_trainer.datamodule.update_rollout
    fake_trainer.sanity_checking = False

    callback.on_validation_epoch_end(fake_trainer, fake_forecaster, None)

    spy.assert_called_once_with(rollout=11)  # Offset 1
