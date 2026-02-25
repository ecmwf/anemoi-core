# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _requested_world_size(default: int = 3) -> int:
    raw = os.getenv("ANEMOI_DISTRIBUTED_TEST_WORLD_SIZE", str(default))
    world_size = int(raw)
    if world_size < 2:
        msg = f"ANEMOI_DISTRIBUTED_TEST_WORLD_SIZE must be >= 2, got {world_size}"
        raise ValueError(msg)
    return world_size


def _run_distributed_strategy_parity(backend: str, world_size: int, suite: str) -> None:
    runner_path = Path(__file__).with_name("distributed_strategy_runner.py")
    cmd = [
        sys.executable,
        str(runner_path),
        "--backend",
        backend,
        "--suite",
        suite,
        "--world-size",
        str(world_size),
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)  # noqa: S603
    if completed.returncode != 0:
        pytest.fail(
            "distributed strategy parity runner failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}",
        )


@pytest.mark.slow
@pytest.mark.multigpu
def test_distributed_strategy_gradient_scaling_diffusion_parity_nccl() -> None:
    world_size = _requested_world_size()
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires at least {world_size} CUDA devices for distributed strategy gradient parity.")
    _run_distributed_strategy_parity(backend="nccl", world_size=world_size, suite="diffusion")


@pytest.mark.slow
@pytest.mark.multigpu
def test_distributed_strategy_ensemble_partition_parity_nccl() -> None:
    world_size = _requested_world_size(default=2)
    if world_size % 2 != 0:
        pytest.skip(
            f"Requires even world size for ensemble partition parity (got {world_size}). "
            "Set ANEMOI_DISTRIBUTED_TEST_WORLD_SIZE=2 (or another even value).",
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires at least {world_size} CUDA devices for distributed strategy ensemble parity.")
    _run_distributed_strategy_parity(backend="nccl", world_size=world_size, suite="ensemble")


@pytest.mark.slow
@pytest.mark.multigpu
def test_distributed_strategy_ensemble_world_step_scaling_nccl() -> None:
    world_size = _requested_world_size(default=2)
    if world_size % 2 != 0:
        pytest.skip(
            f"Requires even world size for ensemble world-step scaling test (got {world_size}). "
            "Set ANEMOI_DISTRIBUTED_TEST_WORLD_SIZE=2 (or another even value).",
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(
            f"Requires at least {world_size} CUDA devices for distributed strategy ensemble world-step scaling.",
        )
    _run_distributed_strategy_parity(backend="nccl", world_size=world_size, suite="ensemble_world_step")
