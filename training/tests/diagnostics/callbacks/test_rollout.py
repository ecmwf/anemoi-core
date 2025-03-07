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
from anemoi.training.schedulers.rollout import InterEpochRolloutMixin
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
    assert hasattr(callback, "on_train_epoch_end")

    return callback


def test_on_load_checkpoint(
    fake_trainer: AnemoiTrainer,
    fake_forecaster: GraphForecaster,
    callback: UpdateRollout,
    checkpoint: dict,
) -> None:
    callback.on_load_checkpoint(fake_trainer, fake_forecaster, checkpoint)
    spy = fake_trainer.datamodule.update_rollout

    spy.assert_called_once_with()


def test_on_train_epoch_end(
    fake_trainer: AnemoiTrainer,
    fake_forecaster: GraphForecaster,
    callback: UpdateRollout,
) -> None:
    fake_trainer.current_epoch = 10
    spy = fake_trainer.datamodule.update_rollout
    fake_trainer.sanity_checking = False

    callback.on_train_epoch_end(fake_trainer, fake_forecaster, None, None)

    spy.assert_called_once()


class DebugInterScheduler(DebugScheduler, InterEpochRolloutMixin):
    pass


def test_inter() -> None:
    sched = DebugInterScheduler(adjust_maximum=10)
    assert sched.current_maximum == 10

    with sched.at(epoch=10):
        assert sched.rollout == 10
        assert sched.current_maximum == 20
