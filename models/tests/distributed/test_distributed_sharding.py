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
    raw = os.getenv("ANEMOI_DISTRIBUTED_TEST_WORLD_SIZE", "2")
    world_size = int(raw)
    if world_size < 2:
        raise ValueError(f"ANEMOI_DISTRIBUTED_TEST_WORLD_SIZE must be >= 2, got {world_size}")
    return world_size


def _skip_if_world_size_exceeds(*, world_size: int, max_world_size: int, test_name: str) -> None:
    if world_size > max_world_size:
        pytest.skip(f"{test_name} supports at most {max_world_size} ranks, got world_size={world_size}.")


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
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    world_size = _requested_world_size()
    _run_spawn_suite(backend=backend, suite="core", nproc_per_node=world_size)


@pytest.mark.multigpu
def test_distributed_graph_primitives_channels_nccl() -> None:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    world_size = _requested_world_size()
    _run_spawn_suite(backend=backend, suite="channels", nproc_per_node=world_size)


@pytest.mark.multigpu
def test_distributed_transformer_sharding_nccl() -> None:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    world_size = _requested_world_size()
    _run_spawn_suite(backend=backend, suite="transformer", nproc_per_node=world_size)


@pytest.mark.multigpu
def test_distributed_transformer_singleton_head_sharding_nccl() -> None:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    world_size = _requested_world_size()
    _skip_if_world_size_exceeds(
        world_size=world_size,
        max_world_size=3,
        test_name="Transformer singleton-head sharding test",
    )
    _run_spawn_suite(backend=backend, suite="transformer_singleton", nproc_per_node=world_size)
