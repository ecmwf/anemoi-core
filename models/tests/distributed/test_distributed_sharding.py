# (C) Copyright 2026 Anemoi contributors.
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


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False

    msg = f"Invalid value for {name}: {raw!r}. Use one of 1/0, true/false, yes/no, on/off."
    raise ValueError(msg)


def _requested_world_size() -> int:
    raw = os.getenv("ANEMOI_DISTRIBUTED_TEST_WORLD_SIZE", "3")
    world_size = int(raw)
    if world_size < 2:
        raise ValueError(f"ANEMOI_DISTRIBUTED_TEST_WORLD_SIZE must be >= 2, got {world_size}")
    return world_size


def _run_spawn_suite(backend: str, suite: str, nproc_per_node: int = 2) -> None:
    worker_path = Path(__file__).with_name("distributed_runner.py")
    cmd = [
        sys.executable,
        str(worker_path),
        "--backend",
        backend,
        "--suite",
        suite,
        "--world-size",
        str(nproc_per_node),
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    deterministic = _env_flag("ANEMOI_DISTRIBUTED_TEST_DETERMINISTIC", default=True)
    env["ANEMOI_DISTRIBUTED_TEST_DETERMINISTIC"] = "1" if deterministic else "0"
    if deterministic and backend == "nccl":
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    completed = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if completed.returncode != 0:
        pytest.fail(
            "distributed worker failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}",
        )


@pytest.mark.multigpu
def test_distributed_graph_primitives_core_nccl() -> None:
    world_size = _requested_world_size()
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires at least {world_size} CUDA devices for NCCL sharding primitives.")
    _run_spawn_suite(backend="nccl", suite="core", nproc_per_node=world_size)


@pytest.mark.multigpu
def test_distributed_graph_primitives_channels_nccl() -> None:
    world_size = _requested_world_size()
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires at least {world_size} CUDA devices for NCCL all_to_all channel sharding.")
    _run_spawn_suite(backend="nccl", suite="channels", nproc_per_node=world_size)


@pytest.mark.multigpu
def test_distributed_transformer_sharding_nccl() -> None:
    world_size = _requested_world_size()
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires at least {world_size} CUDA devices for NCCL transformer sharding.")
    _run_spawn_suite(backend="nccl", suite="transformer", nproc_per_node=world_size)
