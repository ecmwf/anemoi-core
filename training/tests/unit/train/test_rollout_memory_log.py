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
from anemoi.training.train.methods.ensemble import _claim_rollout_memory_log
from anemoi.training.train.methods.ensemble import _log_rollout_memory

GIB = 2**30


@pytest.fixture(autouse=True)
def _fresh_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ensemble, "_rollout_memory_batches_logged", {})
    monkeypatch.delenv(ROLLOUT_MEMORY_ENV, raising=False)


@pytest.fixture
def fake_cuda(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """A CUDA with 80 GiB, whose memory counters the test sets."""
    state = {"allocated": 10 * GIB, "peak": 12 * GIB, "reserved": 14 * GIB}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_: state["allocated"])
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: state["peak"])
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *_: state["reserved"])
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
        _log_rollout_memory(1, "training, 12 steps | step 3 after forward", torch.device("cuda", 1))

    (message,) = [record.getMessage() for record in caplog.records]
    assert message == (
        "rollout memory | rank 1 | training, 12 steps | step 3 after forward | allocated 10.00 GiB | "
        "peak allocated 12.00 GiB | reserved 14.00 GiB | device 80.00 GiB"
    )


def test_a_line_reads_the_given_device_not_the_current_one(monkeypatch: pytest.MonkeyPatch) -> None:
    queried: list[object] = []

    def record(value: int) -> Callable[[object], int]:
        def counter(device: object = None) -> int:
            queried.append(device)
            return value

        return counter

    monkeypatch.setattr(torch.cuda, "memory_allocated", record(GIB))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", record(GIB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", record(GIB))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(total_memory=80 * GIB))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    _log_rollout_memory(1, "validation, 12 steps | step 1 after loss", torch.device("cuda", 1))

    assert queried == [torch.device("cuda", 1)] * 3
