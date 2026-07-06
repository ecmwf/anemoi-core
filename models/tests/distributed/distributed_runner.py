# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Utilities for launching distributed pytest checks.

These functions handle the setup and launching of torch multiprocess tests, including
verification of spawned ranks, backends, and devices. Tests of parallel kernels,
communication primitives should live in separate test modules that call ``run_distributed_test``,
see models/tests/distributed/test_communication_primitives.py for an example.

Set ``ANEMOI_DISTRIBUTED_TEST_DEBUG=1`` and run pytest with ``-s`` to print
rank/backend/device information from each spawned process.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run_distributed_test(
    rank_fn: Callable[..., None],
    *,
    backend: str,
    world_size: int,
    **rank_kwargs: Any,
) -> None:
    """Launch worker processes and run a distributed test function.

    This helper spawns ``world_size`` processes, initializes a process group in
    each, validates the backend/device context, and then calls ``rank_fn`` once
    per rank with ``rank``, ``world_size``, ``device``, ``group``, and any
    forwarded keyword arguments.
    """
    if world_size < 2:
        msg = f"world_size must be >= 2, got {world_size}"
        raise ValueError(msg)
    if backend == "nccl":
        if not torch.cuda.is_available():
            msg = "NCCL backend requested but CUDA is not available."
            raise RuntimeError(msg)
        if torch.cuda.device_count() < world_size:
            msg = (
                f"NCCL backend requested with world_size={world_size}, "
                f"but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )
            raise RuntimeError(msg)

    fd, path = tempfile.mkstemp(prefix="anemoi_dist_init_", suffix=".tmp")
    os.close(fd)
    init_file = Path(path)
    try:
        mp.spawn(
            _run_rank,
            args=(world_size, backend, str(init_file), rank_fn, rank_kwargs),
            nprocs=world_size,
            join=True,
        )
    finally:
        init_file.unlink(missing_ok=True)


def _run_rank(
    rank: int,
    world_size: int,
    backend: str,
    init_file: str,
    rank_fn: Callable[..., None],
    rank_kwargs: dict[str, Any],
) -> None:
    """Spawn entry point that initializes one distributed rank."""
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")

    dist.init_process_group(
        backend=backend,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )

    try:
        group = dist.group.WORLD
        _assert_distributed_context(rank, world_size, backend, device, group)
        rank_fn(rank=rank, world_size=world_size, device=device, group=group, **rank_kwargs)
    finally:
        dist.destroy_process_group()


def _assert_distributed_context(
    rank: int,
    world_size: int,
    backend: str,
    device: torch.device,
    group: dist.ProcessGroup,
) -> None:
    """Verify rank, backend, world size, and device placement."""
    assert dist.is_initialized(), "Expected torch.distributed process group to be initialized."
    assert dist.get_backend(group) == backend, f"Expected backend {backend}, got {dist.get_backend(group)}."
    assert dist.get_world_size(group=group) == world_size, (
        f"Expected world_size={world_size}, got {dist.get_world_size(group=group)}."
    )
    assert dist.get_rank(group=group) == rank, f"Expected rank={rank}, got {dist.get_rank(group=group)}."

    if backend == "nccl":
        assert device.type == "cuda", f"Expected NCCL rank {rank} to use a CUDA device, got {device}."
        assert torch.cuda.is_available(), "Expected CUDA to be available for NCCL tests."
        assert torch.cuda.device_count() >= world_size, (
            f"Expected at least {world_size} CUDA devices, got {torch.cuda.device_count()}."
        )
        assert torch.cuda.current_device() == rank, (
            f"Expected current CUDA device {rank}, got {torch.cuda.current_device()}."
        )
        probe = torch.empty(1, device=device)
        assert probe.is_cuda, "Expected NCCL probe tensor to be a CUDA tensor."
        assert probe.device.index == rank, f"Expected probe on cuda:{rank}, got {probe.device}."
    else:
        assert device.type == "cpu", f"Expected {backend} rank {rank} to use CPU, got {device}."

    if os.getenv("ANEMOI_DISTRIBUTED_TEST_DEBUG", "0") == "1":
        cuda_device = torch.cuda.current_device() if device.type == "cuda" else "n/a"
        print(
            f"rank={rank} world_size={world_size} backend={dist.get_backend(group)} "
            f"device={device} cuda_current_device={cuda_device}",
            flush=True,
        )
