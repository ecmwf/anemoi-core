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

from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import reduce_tensor
from anemoi.models.distributed.graph import shard_tensor

GATHER_TENSOR_CASES = [
    "dim0_even",
    "dim0_uneven",
    "dim1_uneven",
    "negative_dim_uneven",
]
REDUCE_TENSOR_CASES = ["rank_offset"]
SHARD_TENSOR_CASES = GATHER_TENSOR_CASES


def _requested_backend() -> str:
    backend = os.getenv("ANEMOI_DISTRIBUTED_TEST_BACKEND", "gloo").strip().lower()
    if backend not in {"gloo", "nccl"}:
        msg = f"ANEMOI_DISTRIBUTED_TEST_BACKEND must be 'gloo' or 'nccl', got {backend!r}"
        raise ValueError(msg)
    if backend == "nccl" and not torch.cuda.is_available():
        pytest.skip("NCCL backend requested but CUDA is not available.")
    return backend


def _requested_world_size() -> int:
    raw = os.getenv("ANEMOI_DISTRIBUTED_TEST_WORLD_SIZE", "2")
    world_size = int(raw)
    if world_size < 2:
        msg = f"ANEMOI_DISTRIBUTED_TEST_WORLD_SIZE must be >= 2, got {world_size}"
        raise ValueError(msg)
    return world_size


def _run_spawned_primitive(*, backend: str, primitive: str, case_name: str, world_size: int) -> None:
    if backend == "nccl" and torch.cuda.device_count() < world_size:
        pytest.skip(
            f"NCCL backend requested with world_size={world_size}, but only {torch.cuda.device_count()} GPUs found."
        )

    runner_path = Path(__file__).with_name("distributed_runner.py")
    cmd = [
        sys.executable,
        str(runner_path),
        "--backend",
        backend,
        "--primitive",
        primitive,
        "--case",
        case_name,
        "--world-size",
        str(world_size),
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if completed.returncode != 0:
        pytest.fail(
            "distributed communication primitive worker failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


class TestGatherTensorLocal:
    def test_gather_tensor_without_process_group_is_identity_with_identity_backward(self) -> None:
        x = torch.arange(12, dtype=torch.float32).reshape(4, 3).requires_grad_(True)

        y = gather_tensor(x, dim=0, sizes=[x.shape[0]], mgroup=None)

        torch.testing.assert_close(y, x)
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))


class TestShardTensorLocal:
    def test_shard_tensor_without_process_group_is_identity_with_identity_backward(self) -> None:
        x = torch.arange(12, dtype=torch.float32).reshape(4, 3).requires_grad_(True)

        y = shard_tensor(x, dim=0, sizes=[x.shape[0]], mgroup=None)

        torch.testing.assert_close(y, x)
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))


class TestReduceTensorLocal:
    def test_reduce_tensor_without_process_group_is_identity_with_identity_backward(self) -> None:
        x = torch.arange(12, dtype=torch.float32).reshape(4, 3).requires_grad_(True)

        y = reduce_tensor(x, mgroup=None)

        torch.testing.assert_close(y, x)
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))


@pytest.mark.distributed
class TestGatherTensor:
    @pytest.mark.parametrize("case_name", GATHER_TENSOR_CASES, ids=GATHER_TENSOR_CASES)
    def test_gather_tensor_distributed_forward_gathers_and_backward_splits(self, case_name: str) -> None:
        _run_spawned_primitive(
            backend=_requested_backend(),
            primitive="gather_tensor",
            case_name=case_name,
            world_size=_requested_world_size(),
        )


@pytest.mark.distributed
class TestReduceTensor:
    @pytest.mark.parametrize("case_name", REDUCE_TENSOR_CASES, ids=REDUCE_TENSOR_CASES)
    def test_reduce_tensor_distributed_forward_reduces_and_backward_is_identity(self, case_name: str) -> None:
        _run_spawned_primitive(
            backend=_requested_backend(),
            primitive="reduce_tensor",
            case_name=case_name,
            world_size=_requested_world_size(),
        )


@pytest.mark.distributed
class TestShardTensor:
    @pytest.mark.parametrize("case_name", SHARD_TENSOR_CASES, ids=SHARD_TENSOR_CASES)
    def test_shard_tensor_distributed_forward_splits_and_backward_gathers(self, case_name: str) -> None:
        _run_spawned_primitive(
            backend=_requested_backend(),
            primitive="shard_tensor",
            case_name=case_name,
            world_size=_requested_world_size(),
        )
