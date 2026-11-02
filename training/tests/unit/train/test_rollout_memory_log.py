# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The opt-in GPU memory log through a rollout (ANEMOI_LOG_ROLLOUT_MEMORY)."""

import logging
from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch

from anemoi.training.train.methods import ensemble
from anemoi.training.train.methods.ensemble import ROLLOUT_MEMORY_ENV
from anemoi.training.train.methods.ensemble import EnsembleTraining
from anemoi.training.train.methods.ensemble import _claim_rollout_memory_log
from anemoi.training.train.methods.ensemble import _log_rollout_memory
from tests.unit.train.test_rollout_windows import _batch
from tests.unit.train.test_rollout_windows import _loss
from tests.unit.train.test_rollout_windows import _module
from tests.unit.train.test_rollout_windows import _target

GIB = 2**30


@pytest.fixture(autouse=True)
def _fresh_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ensemble, "_rollout_memory_batches_logged", {})
    monkeypatch.delenv(ROLLOUT_MEMORY_ENV, raising=False)


@pytest.fixture
def fake_cuda(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """A CUDA with 80 GiB, whose memory counters the test sets."""
    state = {"peak": 12 * GIB, "peak_reserved": 14 * GIB}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: state["peak"])
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *_: state["peak_reserved"])
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=80 * GIB))
    return state


@pytest.mark.usefixtures("fake_cuda")
def test_it_is_off_unless_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    assert not _claim_rollout_memory_log("training", 12)
    for value in ("0", "-1", "", "three"):
        monkeypatch.setenv(ROLLOUT_MEMORY_ENV, value)
        assert not _claim_rollout_memory_log("training", 12)


def test_it_is_a_no_op_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ROLLOUT_MEMORY_ENV, "3")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not _claim_rollout_memory_log("training", 12)


@pytest.mark.usefixtures("fake_cuda")
def test_the_first_k_batches_of_each_mode_and_rollout_length_are_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ROLLOUT_MEMORY_ENV, "2")

    assert [_claim_rollout_memory_log("training", 1) for _ in range(4)] == [True, True, False, False]
    # a new rollout length (the next epoch) and the other mode count separately
    assert [_claim_rollout_memory_log("training", 2) for _ in range(3)] == [True, True, False]
    assert [_claim_rollout_memory_log("validation", 12) for _ in range(3)] == [True, True, False]
    assert not _claim_rollout_memory_log("training", 1)


@pytest.mark.usefixtures("fake_cuda")
def test_a_line_reports_the_rank_the_label_and_the_memory_in_gib(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger=ensemble.LOGGER.name):
        _log_rollout_memory(0, "training, 12 steps", torch.device("cuda", 1))

    (message,) = [record.getMessage() for record in caplog.records]
    assert message == (
        "rollout memory | rank 0 | training, 12 steps | peak allocated 12.00 GiB | peak reserved 14.00 GiB | "
        "device 80.00 GiB"
    )


def test_a_line_reads_the_given_device_not_the_current_one(monkeypatch: pytest.MonkeyPatch) -> None:
    queried: list[object] = []

    def record(value: int) -> Callable[[object], int]:
        def counter(device: object = None) -> int:
            queried.append(device)
            return value

        return counter

    monkeypatch.setattr(torch.cuda, "max_memory_allocated", record(GIB))
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", record(GIB))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(total_memory=80 * GIB))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    _log_rollout_memory(0, "validation, 12 steps", torch.device("cuda", 1))

    assert queried == [torch.device("cuda", 1)] * 2


@pytest.fixture
def module(monkeypatch: pytest.MonkeyPatch) -> Callable[..., EnsembleTraining]:
    """The rollout-window tests' module (targets from the batch, each step's prediction the next state) on a rank."""

    def make(rollout: int, windows: tuple[int, tuple[float, ...]] | None = None, rank: int = 0) -> EnsembleTraining:
        module = _module(rollout, windows)
        monkeypatch.setattr(module, "compute_loss_metrics", _loss)
        monkeypatch.setattr(
            module.task,
            "get_targets",
            lambda batch, rollout_step=0, **_kw: _target(batch, rollout_step),
        )
        monkeypatch.setattr(module.task, "advance_input", lambda _x, y_pred, *_a, **_kw: dict(y_pred))
        monkeypatch.setattr(EnsembleTraining, "global_rank", property(lambda _self: rank))
        return module

    return make


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.getMessage().startswith("rollout memory")]


@pytest.mark.usefixtures("fake_cuda")
def test_rank_0_logs_one_line_per_training_batch_at_its_end(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    module: Callable[..., EnsembleTraining],
) -> None:
    monkeypatch.setenv(ROLLOUT_MEMORY_ENV, "2")
    forecaster = module(3)
    with caplog.at_level(logging.INFO, logger=ensemble.LOGGER.name):
        for batch_idx in range(3):
            forecaster._step(_batch(3)).loss.backward()
            assert len(_messages(caplog)) == batch_idx  # nothing until the batch has ended
            forecaster.on_train_batch_end(None, None, batch_idx)

    line = "rollout memory | rank 0 | training, 3 steps | peak allocated 12.00 GiB | peak reserved 14.00 GiB | "
    assert _messages(caplog) == [line + "device 80.00 GiB"] * 2


@pytest.mark.usefixtures("fake_cuda")
def test_a_validation_batch_logs_one_line_after_its_last_step(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    module: Callable[..., EnsembleTraining],
) -> None:
    monkeypatch.setenv(ROLLOUT_MEMORY_ENV, "2")
    forecaster = module(3)
    with caplog.at_level(logging.INFO, logger=ensemble.LOGGER.name), torch.no_grad():
        for _ in range(3):
            forecaster._step(_batch(3), validation_mode=True)

    assert [message.split(" | peak")[0] for message in _messages(caplog)] == [
        "rollout memory | rank 0 | validation, 3 steps",
    ] * 2


@pytest.mark.usefixtures("fake_cuda")
def test_a_windowed_batch_logs_one_line_at_its_end(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    module: Callable[..., EnsembleTraining],
) -> None:
    monkeypatch.setenv(ROLLOUT_MEMORY_ENV, "1")
    forecaster = module(4, (2, (1.0, 1.0)))
    with caplog.at_level(logging.INFO, logger=ensemble.LOGGER.name):
        forecaster._windowed_rollout_backward(_batch(4), lambda loss, _last: loss.backward())
        assert _messages(caplog) == []
        forecaster.on_train_batch_end(None, None, 0)

    assert [message.split(" | peak")[0] for message in _messages(caplog)] == [
        "rollout memory | rank 0 | training, 4 steps in 2 windows",
    ]


@pytest.mark.usefixtures("fake_cuda")
def test_the_other_ranks_log_nothing(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    module: Callable[..., EnsembleTraining],
) -> None:
    monkeypatch.setenv(ROLLOUT_MEMORY_ENV, "3")
    forecaster = module(3, rank=1)
    windowed = module(4, (2, (1.0, 1.0)), rank=1)
    with caplog.at_level(logging.INFO, logger=ensemble.LOGGER.name):
        forecaster._step(_batch(3)).loss.backward()
        forecaster.on_train_batch_end(None, None, 0)
        with torch.no_grad():
            forecaster._step(_batch(3), validation_mode=True)
        windowed._windowed_rollout_backward(_batch(4), lambda loss, _last: loss.backward())
        windowed.on_train_batch_end(None, None, 0)

    assert _messages(caplog) == []
