# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from pytorch_lightning import Trainer

from anemoi.training.diagnostics.callbacks.stopping import TimeLimit


@pytest.fixture
def mock_trainer() -> Mock:
    training_process = Mock(Trainer)
    training_process.should_stop = False
    return training_process


def test_time_limit_stops_training(mock_trainer: Mock) -> None:
    # Setting a time limit of 1 second
    time_limit_callback = TimeLimit(config=None, limit="24h")

    # Mocking the time to simulate training for 2 seconds
    with patch("time.time", return_value=time_limit_callback._start_time + 24 * 60 * 60 + 1):
        time_limit_callback.on_train_epoch_end(mock_trainer, None)

    # Check if the training process should stop
    assert mock_trainer.should_stop


def test_time_limit_stops_training_int(mock_trainer: Mock) -> None:
    # Setting a time limit of 1 second
    time_limit_callback = TimeLimit(config=None, limit=24)

    # Mocking the time to simulate training for 2 seconds
    with patch("time.time", return_value=time_limit_callback._start_time + 24 * 60 * 60 + 1):
        time_limit_callback.on_train_epoch_end(mock_trainer, None)

    # Check if the training process should stop
    assert mock_trainer.should_stop


def test_time_limit_does_not_stop_training(mock_trainer: Mock) -> None:
    # Setting a time limit of 1 hour
    time_limit_callback = TimeLimit(config=None, limit="1h")

    # Mocking the time to simulate training for 2 seconds
    with patch("time.time", return_value=time_limit_callback._start_time + 2):
        time_limit_callback.on_train_epoch_end(mock_trainer, None)

    # Check if the training process should not stop
    assert not mock_trainer.should_stop


def test_time_limit_creates_file_on_stop(mock_trainer: Mock, tmp_path: Any) -> None:
    # Setting a time limit of 1 second
    time_limit_callback = TimeLimit(config=None, limit="1s", record_file=tmp_path / "log")

    # Mocking the time to simulate training for 2 seconds
    with patch("time.time", return_value=time_limit_callback._start_time + 2):
        time_limit_callback.on_train_epoch_end(mock_trainer, None)

    # Check if the training process should stop
    assert mock_trainer.should_stop

    # Check if the file is created
    assert (tmp_path / "log").exists()


def test_time_limit_does_not_create_file_when_not_stopping(mock_trainer: Mock, tmp_path: Any) -> None:
    # Setting a time limit of 1 second
    time_limit_callback = TimeLimit(config=None, limit="24h", record_file=tmp_path / "log")

    # Mocking the time to simulate training for 2 seconds
    with patch("time.time", return_value=time_limit_callback._start_time + 2):
        time_limit_callback.on_train_epoch_end(mock_trainer, None)

    # Check if the training process should stop
    assert not mock_trainer.should_stop

    # Check if the file is created
    assert not (tmp_path / "log").exists()
