# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Distributed tests for communication primitives.

Tests the low-level tensor communication primitives in
``anemoi.models.distributed.primitives`` across distributed ranks and supported
backends.

These tests are skipped by default. Pass ``--distributed`` to run them. Use
``--distributed-backend`` and ``--distributed-world-size`` to select the backend
and rank count.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.distributed as dist

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.primitives import _alltoall_transpose
from anemoi.models.distributed.primitives import _expand_sharded_tensor
from anemoi.models.distributed.primitives import _gather
from anemoi.models.distributed.primitives import _reduce
from anemoi.models.distributed.primitives import _split
from distributed_runner import run_distributed_test

GLOBAL_DEFAULT_ATOL = 1e-12
GLOBAL_DEFAULT_RTOL = 1e-12

PARTITION_CASES = [
    pytest.param((4, 3), 0, id="dim0_even"),
    pytest.param((5, 3), 0, id="dim0_uneven"),
    pytest.param((3, 5), 1, id="dim1_uneven"),
    pytest.param((3, 5), -1, id="negative_dim_uneven"),
]

REDUCE_CASES = [
    pytest.param(True, id="fp32_path_float32_input"),
    pytest.param(False, id="native_float32_accumulation"),
]

LOW_PRECISION_REDUCE_CASES = [
    pytest.param((2, 3), torch.float16, id="fp32_accumulation_from_fp16"),
    pytest.param((2, 3), torch.bfloat16, id="fp32_accumulation_from_bf16"),
]

ALLTOALL_TRANSPOSE_CASES = [
    pytest.param((4, 6), 0, 1, id="dim0_to_dim1_even"),
    pytest.param((4, 6), 1, 0, id="dim1_to_dim0_even"),
    pytest.param((5, 7), 0, 1, id="dim0_to_dim1_uneven"),
    pytest.param((5, 7), -2, -1, id="negative_dims_uneven"),
    pytest.param((4, 5, 6), 1, 2, id="3d_dim1_to_dim2_uneven"),
]


def _torch_version_less_than(major: int, minor: int) -> bool:
    # TODO: Move to shared util or remove.
    version_parts = torch.__version__.split("+", maxsplit=1)[0].split(".")
    return (int(version_parts[0]), int(version_parts[1])) < (major, minor)


def _test_split_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    shape: tuple[int, ...],
    dim: int,
    atol: float = GLOBAL_DEFAULT_ATOL,
    rtol: float = GLOBAL_DEFAULT_RTOL,
) -> None:
    full = torch.arange(math.prod(shape), dtype=torch.float64, device=device).reshape(shape)
    sizes = get_balanced_partition_sizes(shape[dim], world_size)

    actual = _split(full, dim_=dim, sizes_=sizes, group=group)
    expected = torch.split(full, sizes, dim=dim)[rank].contiguous()

    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.distributed
@pytest.mark.parametrize(("shape", "dim"), PARTITION_CASES)
def test_split_distributes_full_tensor_to_rank_local_slice(
    shape: tuple[int, ...], dim: int, distributed_backend: str, distributed_world_size: int
) -> None:
    run_distributed_test(
        _test_split_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=shape,
        dim=dim,
    )


def _test_gather_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    shape: tuple[int, ...],
    dim: int,
    atol: float = GLOBAL_DEFAULT_ATOL,
    rtol: float = GLOBAL_DEFAULT_RTOL,
) -> None:
    full = torch.arange(math.prod(shape), dtype=torch.float64, device=device).reshape(shape)
    sizes = get_balanced_partition_sizes(shape[dim], world_size)
    local = torch.split(full, sizes, dim=dim)[rank].contiguous()

    actual = _gather(local, dim_=dim, sizes=sizes, group=group)

    torch.testing.assert_close(actual, full, atol=atol, rtol=rtol)


@pytest.mark.distributed
@pytest.mark.parametrize(("shape", "dim"), PARTITION_CASES)
def test_gather_reconstructs_full_tensor_from_rank_local_slices(
    shape: tuple[int, ...], dim: int, distributed_backend: str, distributed_world_size: int
) -> None:
    run_distributed_test(
        _test_gather_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=shape,
        dim=dim,
    )


def _test_reduce_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    shape: tuple[int, ...],
    use_fp32: bool,
    dtype: torch.dtype = torch.float32,
    atol: float = GLOBAL_DEFAULT_ATOL,
    rtol: float = GLOBAL_DEFAULT_RTOL,
) -> None:
    base = torch.arange(math.prod(shape), dtype=dtype, device=device).reshape(shape)
    local = base + rank

    actual = _reduce(local, use_fp32=use_fp32, group=group)
    expected = sum(base + other for other in range(world_size))

    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.distributed
@pytest.mark.parametrize("use_fp32", REDUCE_CASES)
def test_reduce_sums_same_shape_rank_local_tensors(
    use_fp32: bool, distributed_backend: str, distributed_world_size: int
) -> None:
    run_distributed_test(
        _test_reduce_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=(2, 3),
        use_fp32=use_fp32,
    )


@pytest.mark.distributed
@pytest.mark.parametrize(("shape", "dtype"), LOW_PRECISION_REDUCE_CASES)
def test_reduce_fp32_accumulation_supports_low_precision_inputs(
    shape: tuple[int, ...], dtype: torch.dtype, distributed_backend: str, distributed_world_size: int
) -> None:
    if distributed_backend != "nccl":
        pytest.skip("Low-precision fp32 accumulation is only validated on the NCCL/CUDA path.")
    run_distributed_test(
        _test_reduce_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=shape,
        use_fp32=True,
        dtype=dtype,
        atol=1e-2,
        rtol=1e-2,
    )


def _test_expand_sharded_tensor_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    shape: tuple[int, ...],
    dim: int,
    atol: float = GLOBAL_DEFAULT_ATOL,
    rtol: float = GLOBAL_DEFAULT_RTOL,
) -> None:
    full = torch.arange(math.prod(shape), dtype=torch.float64, device=device).reshape(shape)
    sizes = get_balanced_partition_sizes(shape[dim], world_size)
    local = torch.split(full, sizes, dim=dim)[rank].contiguous().clone()

    actual = _expand_sharded_tensor(local, dim_=dim, sizes=sizes, group=group)

    assert actual.shape == full.shape
    normalized_dim = dim % full.dim()
    start = sum(sizes[:rank])
    actual_local_slice = actual.narrow(normalized_dim, start, sizes[rank])
    torch.testing.assert_close(actual_local_slice, local, atol=atol, rtol=rtol)


@pytest.mark.distributed
@pytest.mark.parametrize(("shape", "dim"), PARTITION_CASES)
def test_expand_sharded_tensor_populates_only_rank_local_slice(
    shape: tuple[int, ...], dim: int, distributed_backend: str, distributed_world_size: int
) -> None:
    run_distributed_test(
        _test_expand_sharded_tensor_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=shape,
        dim=dim,
    )


def _test_alltoall_transpose_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    shape: tuple[int, ...],
    dim_split: int,
    dim_concat: int,
    atol: float = GLOBAL_DEFAULT_ATOL,
    rtol: float = GLOBAL_DEFAULT_RTOL,
) -> None:
    full = torch.arange(math.prod(shape), dtype=torch.float64, device=device).reshape(shape)
    ndim = len(shape)
    split_sizes = get_balanced_partition_sizes(shape[dim_split % ndim], world_size)
    concat_sizes = get_balanced_partition_sizes(shape[dim_concat % ndim], world_size)
    local = torch.split(full, concat_sizes, dim=dim_concat)[rank].contiguous()

    actual = _alltoall_transpose(
        local,
        dim_split=dim_split,
        split_sizes=split_sizes,
        dim_concat=dim_concat,
        concat_sizes=concat_sizes,
        group=group,
    )
    expected = torch.split(full, split_sizes, dim=dim_split)[rank].contiguous()

    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.distributed
@pytest.mark.parametrize(("shape", "dim_split", "dim_concat"), ALLTOALL_TRANSPOSE_CASES)
def test_alltoall_transpose_redistributes_between_sharded_layouts(
    shape: tuple[int, ...],
    dim_split: int,
    dim_concat: int,
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    if distributed_backend == "gloo" and _torch_version_less_than(2, 6):
        pytest.skip("Gloo alltoall_transpose requires torch >= 2.6.")
    run_distributed_test(
        _test_alltoall_transpose_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=shape,
        dim_split=dim_split,
        dim_concat=dim_concat,
    )
