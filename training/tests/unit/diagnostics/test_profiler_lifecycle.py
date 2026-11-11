# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch
from omegaconf import DictConfig
from pytorch_lightning.profilers import Profiler
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.utilities import rank_zero_only

from anemoi.training.diagnostics.profilers import BenchmarkProfiler
from anemoi.training.train.profiler import AnemoiProfiler

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


@pytest.fixture
def benchmark_profiler(tmp_path: Path) -> BenchmarkProfiler:
    """Use the production wrapper with only the fields needed for capture construction."""
    return BenchmarkProfiler(
        DictConfig(
            {
                "diagnostics": {
                    "benchmark_profiler": {
                        "memory": {
                            "enabled": True,
                            "warmup": 0,
                            "steps": 2,
                            "extra_plots": False,
                            "trace_rank0_only": False,
                        },
                        "time": {"enabled": True},
                    },
                },
                "system": {"output": {"profiler": str(tmp_path)}},
                "training": {"num_sanity_val_steps": 0},
            },
        ),
    )


@pytest.mark.parametrize("state", ["partial", "time_only", "dummy", "unstarted"])
def test_teardown_is_safe_before_capture_starts(
    benchmark_profiler: BenchmarkProfiler,
    state: str,
) -> None:
    """Missing children, placeholder classes and inactive real children can all be finalized twice."""
    if state in {"partial", "time_only"}:
        outer = BenchmarkProfiler.__new__(BenchmarkProfiler)
        Profiler.__init__(outer)
        if state == "time_only":
            outer.time_profiler = SimpleProfiler()
    else:
        outer = benchmark_profiler
        if state == "unstarted":
            outer.memory_profiler = PyTorchProfiler(
                activities=[torch.profiler.ProfilerActivity.CPU],
                schedule=None,
                record_module_names=False,
                export_to_chrome=False,
            )

    outer.teardown(stage=None)
    outer.teardown(stage=None)

    assert outer._output_file is None
    if state == "unstarted":
        assert outer.memory_profiler.profiler is None
        assert outer.memory_profiler.function_events is None
        assert outer.get_memory_profiler_df() == ""


def test_teardown_closes_other_resources_when_memory_cleanup_fails(
    benchmark_profiler: BenchmarkProfiler,
    mocker: MockerFixture,
) -> None:
    """A child failure cannot leave the sibling profiler or the outer output stream open."""
    memory = mocker.Mock(spec=PyTorchProfiler)
    memory.profiler = None
    failure = RuntimeError("inner close failed")
    memory.teardown.side_effect = failure
    time = mocker.Mock(spec=SimpleProfiler)
    output = mocker.Mock()
    benchmark_profiler.memory_profiler = memory
    benchmark_profiler.time_profiler = time
    benchmark_profiler._output_file = output

    try:
        with pytest.raises(RuntimeError, match="inner close failed") as raised:
            benchmark_profiler.teardown(stage="fit")

        assert raised.value is failure
        time.teardown.assert_called_once_with(stage="fit")
        output.close.assert_called_once_with()
        assert benchmark_profiler._output_file is None
    finally:
        # Do not leave a deliberately failing child for the base destructor to retry.
        memory.teardown.side_effect = None


def test_teardown_is_process_local_not_rank_zero_only(
    benchmark_profiler: BenchmarkProfiler,
    mocker: MockerFixture,
) -> None:
    """Finalization must still run when rank-zero reporting and distributed setup are unavailable."""
    memory = mocker.Mock(spec=PyTorchProfiler)
    memory.profiler = None
    benchmark_profiler.memory_profiler = memory
    mocker.patch.object(rank_zero_only, "rank", 1)
    broadcast = mocker.patch(
        "anemoi.training.diagnostics.profilers.dist.broadcast_object_list",
        side_effect=AssertionError("Cleanup attempted a collective"),
    )

    benchmark_profiler.teardown(stage="fit")

    memory.teardown.assert_called_once_with(stage="fit")
    broadcast.assert_not_called()


def test_extra_plots_use_retained_capture_after_repeated_teardown(
    benchmark_profiler: BenchmarkProfiler,
    mocker: MockerFixture,
) -> None:
    """Legacy plot consumers keep the completed capture after Lightning releases its live reference."""
    benchmark_profiler.config.diagnostics.benchmark_profiler.memory.extra_plots = True
    memory = mocker.Mock(spec=PyTorchProfiler)
    capture = mocker.Mock(spec=["export_memory_timeline"])
    memory.profiler = capture
    benchmark_profiler.memory_profiler = memory

    def close_capture(stage: str | None) -> None:
        del stage
        memory.profiler = None

    memory.teardown.side_effect = close_capture
    plot = mocker.patch("torch.cuda._memory_viz.profile_plot", return_value="<html>memory</html>")

    benchmark_profiler.teardown(stage="fit")
    benchmark_profiler.teardown(stage="fit")
    assert memory.profiler is None
    assert benchmark_profiler._memory_capture is capture

    benchmark_profiler._save_extra_plots()

    plot.assert_called_once_with(capture)
    capture.export_memory_timeline.assert_called_once_with(benchmark_profiler.memory_timeline_fname)
    assert benchmark_profiler.memory_trace_fname.read_text() == "<html>memory</html>"


@pytest.mark.gpu
@pytest.mark.parametrize("training_fails", [False, True], ids=["normal", "training_error"])
def test_application_owner_closes_real_capture_and_retains_events(
    benchmark_profiler: BenchmarkProfiler,
    tmp_path: Path,
    mocker: MockerFixture,
    training_fails: bool,
) -> None:
    """Normal Lightning teardown and exceptional owner cleanup both preserve real CPU/CUDA events."""
    if not torch.cuda.is_available():
        pytest.skip("A CUDA device is required for real profiler lifecycle coverage")

    # Isolate only distributed bootstrap and the existing global patch, not capture/finalization.
    mocker.patch.object(benchmark_profiler, "broadcast_profiler_path", return_value=str(tmp_path))
    mocker.patch("torch.profiler.profile", torch.profiler.profile)
    benchmark_profiler.setup(stage="fit", local_rank=0)
    inner = benchmark_profiler.memory_profiler
    owner = AnemoiProfiler.__new__(AnemoiProfiler)
    owner.profiler = benchmark_profiler
    report = mocker.patch.object(owner, "report")
    export = mocker.patch.object(owner, "export_to_logger")
    failure = RuntimeError("injected training failure")

    def train() -> None:
        with benchmark_profiler.profile("[Strategy]SingleDeviceStrategy.training_step"):
            with torch.profiler.record_function("prof005_completed"):
                data = torch.ones((4, 4), device="cuda")
                result = data @ data
                torch.cuda.synchronize()
            torch.testing.assert_close(result, torch.full_like(result, 4), rtol=0, atol=0)
            if training_fails:
                raise failure
        # Model the normal Lightning owner path; exceptional exits do not reach describe().
        benchmark_profiler.describe()

    mocker.patch.object(owner, "train", side_effect=train)

    try:
        if training_fails:
            with pytest.raises(RuntimeError, match="injected training failure") as raised:
                owner.profile()
            assert raised.value is failure
            report.assert_not_called()
            export.assert_not_called()
        else:
            owner.profile()
            report.assert_called_once_with()
            export.assert_called_once_with()

        # Assert owner-driven closure before any direct report read or test-only cleanup.
        assert inner.profiler is None
        assert not inner._recording_map
        events = inner.function_events
        assert events
        assert any(event.name == "prof005_completed" for event in events)
        traces = list(tmp_path.glob("*.pt.trace.json"))
        assert len(traces) == 1
        trace_events = json.loads(traces[0].read_text())["traceEvents"]
        assert any(event.get("cat") == "kernel" for event in trace_events)
        assert any(
            event.get("cat") == "user_annotation" and event.get("name") == "prof005_completed"
            for event in trace_events
        )

        memory_report = benchmark_profiler.get_memory_profiler_df()
        assert not memory_report.empty
        assert benchmark_profiler.memory_report_fname.exists()
        assert inner.function_events is events
        assert benchmark_profiler.time_profiler.recorded_durations
        benchmark_profiler.teardown(stage="fit")
        assert inner.function_events is events
        assert len(list(tmp_path.glob("*.pt.trace.json"))) == 1
    finally:
        # Test isolation only, after assertions: a regression must fail before any salvage cleanup.
        inner.teardown(stage="fit")
