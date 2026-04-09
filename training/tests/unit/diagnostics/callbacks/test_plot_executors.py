# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# ruff: noqa: ANN001, ANN201

import time
from unittest.mock import MagicMock

import pytest

from anemoi.training.diagnostics.callbacks.plot import AsyncPlotExecutor
from anemoi.training.diagnostics.callbacks.plot import BasePlotExecutor
from anemoi.training.diagnostics.callbacks.plot import SyncPlotExecutor


def test_base_plot_executor_is_abstract():
    """BasePlotExecutor cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BasePlotExecutor()


def test_base_plot_executor_requires_schedule_and_shutdown():
    """A concrete subclass that omits either abstract method is still abstract."""

    class MissingShutdown(BasePlotExecutor):
        def schedule(self, fn, trainer, *args, **kwargs) -> None:
            pass

    class MissingSchedule(BasePlotExecutor):
        def shutdown(self) -> None:
            pass

    with pytest.raises(TypeError):
        MissingShutdown()

    with pytest.raises(TypeError):
        MissingSchedule()


def test_sync_executor_calls_fn_immediately():
    """SyncPlotExecutor invokes fn synchronously before schedule() returns."""
    called_with = {}

    def fn(trainer, args, kwargs) -> None:
        called_with["trainer"] = trainer
        called_with["args"] = args
        called_with["kwargs"] = kwargs

    trainer = MagicMock()
    executor = SyncPlotExecutor()
    executor.schedule(fn, trainer, "a", "b", key="val")

    assert called_with["trainer"] is trainer
    assert called_with["args"] == ("a", "b")
    assert called_with["kwargs"] == {"key": "val"}


def test_sync_executor_shutdown_is_noop():
    """SyncPlotExecutor.shutdown() completes without error."""
    SyncPlotExecutor().shutdown()


def test_sync_executor_is_concrete_base_plot_executor():
    """SyncPlotExecutor satisfies the BasePlotExecutor interface."""
    assert isinstance(SyncPlotExecutor(), BasePlotExecutor)


@pytest.fixture
def async_executor():
    executor = AsyncPlotExecutor()
    yield executor
    executor.shutdown()


def test_async_executor_is_concrete_base_plot_executor(async_executor):
    """AsyncPlotExecutor satisfies the BasePlotExecutor interface."""
    assert isinstance(async_executor, BasePlotExecutor)


def test_async_executor_starts_event_loop(async_executor):
    """AsyncPlotExecutor starts its background event loop on construction."""
    assert async_executor.loop is not None
    assert async_executor.loop.is_running()


def test_async_executor_runs_fn_in_background(async_executor):
    """schedule() runs fn asynchronously and completes within a reasonable time."""
    import threading

    done = threading.Event()

    def fn(_trainer, _args, _kwargs) -> None:
        done.set()

    async_executor.schedule(fn, MagicMock())
    assert done.wait(timeout=5), "fn was not called within 5 seconds"


def test_async_executor_passes_args_and_kwargs(async_executor):
    """schedule() forwards positional args and keyword args to fn correctly."""
    import threading

    received = {}
    done = threading.Event()

    def fn(_trainer, args, kwargs) -> None:
        received["args"] = args
        received["kwargs"] = kwargs
        done.set()

    trainer = MagicMock()
    async_executor.schedule(fn, trainer, 1, 2, x=3)
    assert done.wait(timeout=5)
    assert received["args"] == (1, 2)
    assert received["kwargs"] == {"x": 3}


def test_async_executor_shuts_down_on_fn_exception():
    """An exception raised inside fn triggers executor shutdown (loop stops, resources released)."""
    import threading

    raised = threading.Event()

    msg = "deliberate failure"

    def failing_fn(_trainer, _args, _kwargs) -> None:
        raised.set()
        raise RuntimeError(msg)

    executor = AsyncPlotExecutor()
    executor.schedule(failing_fn, MagicMock())
    assert raised.wait(timeout=5), "failing_fn was not called"

    # Give the shutdown triggered inside _submit time to complete.
    deadline = time.monotonic() + 5.0
    while executor.loop.is_running() and time.monotonic() < deadline:
        time.sleep(0.05)

    assert not executor.loop.is_running(), "executor loop should have stopped after fn raised"


def test_async_executor_shutdown_stops_loop():
    """After shutdown(), the event loop is no longer running."""
    executor = AsyncPlotExecutor()
    executor.shutdown()
    assert not executor.loop.is_running()
