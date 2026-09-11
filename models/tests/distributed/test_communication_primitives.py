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

import pytest
import torch
import torch.distributed as dist
from distributed_runner import run_distributed_test
from distributed_test_utils import shard_sizes_from_pattern
from distributed_test_utils import torch_version_less_than

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.primitives import _alltoall_transpose
from anemoi.models.distributed.primitives import _alltoallwrapper
from anemoi.models.distributed.primitives import _expand_sharded_tensor
from anemoi.models.distributed.primitives import _gather
from anemoi.models.distributed.primitives import _reduce
from anemoi.models.distributed.primitives import _split

GLOBAL_DEFAULT_ATOL = 1e-12
GLOBAL_DEFAULT_RTOL = 1e-12


def _test_split_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    shape: tuple[int, ...],
    dim: int,
    shard_sizes: list[int] | None = None,
    atol: float = GLOBAL_DEFAULT_ATOL,
    rtol: float = GLOBAL_DEFAULT_RTOL,
) -> None:
    full = torch.arange(torch.Size(shape).numel(), dtype=torch.float32, device=device).reshape(shape)
    if shard_sizes is None:
        shard_sizes = get_balanced_partition_sizes(full.size(dim), world_size)

    actual = _split(full, dim_=dim, sizes_=shard_sizes, group=group)
    expected = torch.split(full, shard_sizes, dim=dim)[rank].contiguous()

    assert actual.size() == expected.size()
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("shape", "dim"),
    [
        pytest.param((1024, 1024), 0, id="dim0_even_1m"),
        pytest.param((1025, 1024), 0, id="dim0_uneven_1m"),
        pytest.param((1024, 1025), 1, id="dim1_uneven_1m"),
        pytest.param((64, 129, 128), 1, id="3d_middle_dim_1m"),
        pytest.param((64, 128, 128), -1, id="negative_last_dim_even_1m"),
    ],
)
def test_split_distributes_full_tensor_to_rank_local_slice(
    shape: tuple[int, ...],
    dim: int,
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    run_distributed_test(
        _test_split_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=shape,
        dim=dim,
    )


@pytest.mark.distributed
@pytest.mark.parametrize(
    "shard_size_pattern",
    [
        pytest.param((128, 384, 256, 512), id="dim=0,irregular-shard-sizes=[128,384,256,512]"),
        pytest.param((256, 0, 256, 256), id="dim=0,empty-shard-sizes=[256,0,256,256]"),
    ],
)
def test_split_supports_explicit_shard_sizes(
    shard_size_pattern: tuple[int, ...],
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    shard_sizes = shard_sizes_from_pattern(shard_size_pattern, distributed_world_size)
    run_distributed_test(
        _test_split_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=(sum(shard_sizes), 1024),
        dim=0,
        shard_sizes=shard_sizes,
    )


def _test_gather_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    shape: tuple[int, ...],
    dim: int,
    shard_sizes: list[int] | None = None,
    atol: float = GLOBAL_DEFAULT_ATOL,
    rtol: float = GLOBAL_DEFAULT_RTOL,
) -> None:
    full = torch.arange(torch.Size(shape).numel(), dtype=torch.float32, device=device).reshape(shape)
    if shard_sizes is None:
        shard_sizes = get_balanced_partition_sizes(full.size(dim), world_size)
    local = torch.split(full, shard_sizes, dim=dim)[rank].contiguous()

    actual = _gather(local, dim_=dim, sizes=shard_sizes, group=group)

    assert actual.size() == full.size()
    assert actual.dtype == full.dtype
    assert actual.device == full.device
    torch.testing.assert_close(actual, full, atol=atol, rtol=rtol)


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("shape", "dim"),
    [
        pytest.param((1024, 1024), 0, id="dim0_even_into_tensor_1m"),
        pytest.param((1025, 1024), 0, id="dim0_uneven_padding_1m"),
        pytest.param((1024, 1024), 1, id="dim1_even_default_1m"),
        pytest.param((1024, 1025), 1, id="dim1_uneven_padding_1m"),
        pytest.param((64, 128, 128), -1, id="negative_last_dim_even_1m"),
    ],
)
def test_gather_reconstructs_full_tensor_from_rank_local_slices(
    shape: tuple[int, ...],
    dim: int,
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    run_distributed_test(
        _test_gather_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=shape,
        dim=dim,
    )


@pytest.mark.distributed
@pytest.mark.parametrize(
    "shard_size_pattern",
    [
        pytest.param((128, 384, 256, 512), id="dim=0,irregular-shard-sizes=[128,384,256,512]"),
        pytest.param((256, 0, 256, 256), id="dim=0,empty-shard-sizes=[256,0,256,256]"),
    ],
)
def test_gather_supports_explicit_shard_sizes(
    shard_size_pattern: tuple[int, ...],
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    shard_sizes = shard_sizes_from_pattern(shard_size_pattern, distributed_world_size)
    run_distributed_test(
        _test_gather_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=(sum(shard_sizes), 1024),
        dim=0,
        shard_sizes=shard_sizes,
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
    if dtype == torch.float32:
        base = torch.arange(torch.Size(shape).numel(), dtype=dtype, device=device).reshape(shape)
    else:
        # Keep low-precision values bounded: large arange values overflow float16
        # and are heavily quantized in bfloat16. Build in float32, then cast so
        # the reduce input still has the requested dtype.
        base = torch.linspace(0, 1, torch.Size(shape).numel(), dtype=torch.float32, device=device).reshape(shape)
        base = base.to(dtype=dtype)

    local = base + rank
    actual = _reduce(local, use_fp32=use_fp32, group=group)

    # Build the conceptual per-rank inputs as a leading "rank" dimension:
    # inputs[q] == base + q. Summing over dim=0 gives the all-reduce result.
    rank_offsets = torch.arange(world_size, dtype=dtype, device=device).reshape(world_size, *([1] * base.ndim))
    inputs = base.unsqueeze(0) + rank_offsets
    if use_fp32:
        expected = torch.sum(inputs.float(), dim=0).to(dtype)
    else:
        expected = torch.sum(inputs, dim=0)

    assert actual.size() == expected.size()
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("shape", "dtype", "use_fp32"),
    [
        pytest.param((1024, 1024), torch.float32, True, id="float32_fp32_path_1m"),
        pytest.param((1024, 1024), torch.float32, False, id="float32_native_path_1m"),
        pytest.param((64, 128, 129), torch.float32, True, id="float32_fp32_path_3d_1m"),
        pytest.param((64, 128, 129), torch.float32, False, id="float32_native_path_3d_1m"),
    ],
)
def test_reduce_sums_same_shape_rank_local_tensors(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    use_fp32: bool,
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    run_distributed_test(
        _test_reduce_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=shape,
        use_fp32=use_fp32,
        dtype=dtype,
    )


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        pytest.param((1024, 1024), torch.float16, id="fp32_accumulation_from_fp16_1m"),
        pytest.param((1024, 1024), torch.bfloat16, id="fp32_accumulation_from_bf16_1m"),
    ],
)
def test_reduce_fp32_accumulation_supports_low_precision_inputs(
    shape: tuple[int, ...], dtype: torch.dtype, distributed_backend: str, distributed_world_size: int
) -> None:
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
    shard_sizes: list[int] | None = None,
    atol: float = GLOBAL_DEFAULT_ATOL,
    rtol: float = GLOBAL_DEFAULT_RTOL,
) -> None:
    full = torch.arange(torch.Size(shape).numel(), dtype=torch.float32, device=device).reshape(shape)
    if shard_sizes is None:
        shard_sizes = get_balanced_partition_sizes(full.size(dim), world_size)
    local = torch.split(full, shard_sizes, dim=dim)[rank].contiguous().clone()

    actual = _expand_sharded_tensor(local, dim_=dim, sizes=shard_sizes, group=group)

    assert actual.size() == full.size()
    assert actual.dtype == full.dtype
    assert actual.device == full.device

    normalized_dim = dim % full.dim()
    start = sum(shard_sizes[:rank])
    actual_local_slice = actual.narrow(normalized_dim, start, shard_sizes[rank])

    assert actual_local_slice.size() == local.size()
    assert actual_local_slice.dtype == local.dtype
    assert actual_local_slice.device == local.device
    torch.testing.assert_close(actual_local_slice, local, atol=atol, rtol=rtol)


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("shape", "dim"),
    [
        pytest.param((1025, 1024), 0, id="dim0_uneven_1m"),
        pytest.param((1024, 1025), 1, id="dim1_uneven_1m"),
        pytest.param((64, 129, 128), 1, id="3d_middle_dim_1m"),
        pytest.param((64, 128, 128), -1, id="negative_last_dim_even_1m"),
    ],
)
def test_expand_sharded_tensor_populates_only_rank_local_slice(
    shape: tuple[int, ...],
    dim: int,
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    run_distributed_test(
        _test_expand_sharded_tensor_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=shape,
        dim=dim,
    )


@pytest.mark.distributed
@pytest.mark.parametrize(
    "shard_size_pattern",
    [
        pytest.param((128, 384, 256, 512), id="dim=0,irregular-shard-sizes=[128,384,256,512]"),
        pytest.param((256, 0, 256, 256), id="dim=0,empty-shard-sizes=[256,0,256,256]"),
    ],
)
def test_expand_sharded_tensor_supports_explicit_shard_sizes(
    shard_size_pattern: tuple[int, ...],
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    shard_sizes = shard_sizes_from_pattern(shard_size_pattern, distributed_world_size)
    run_distributed_test(
        _test_expand_sharded_tensor_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=(sum(shard_sizes), 1024),
        dim=0,
        shard_sizes=shard_sizes,
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
    split_shard_sizes: list[int] | None = None,
    concat_shard_sizes: list[int] | None = None,
    atol: float = GLOBAL_DEFAULT_ATOL,
    rtol: float = GLOBAL_DEFAULT_RTOL,
) -> None:
    full = torch.arange(torch.Size(shape).numel(), dtype=torch.float32, device=device).reshape(shape)
    if split_shard_sizes is None:
        split_shard_sizes = get_balanced_partition_sizes(full.size(dim_split), world_size)
    if concat_shard_sizes is None:
        concat_shard_sizes = get_balanced_partition_sizes(full.size(dim_concat), world_size)
    local = torch.split(full, concat_shard_sizes, dim=dim_concat)[rank].contiguous()

    actual = _alltoall_transpose(
        local,
        dim_split=dim_split,
        split_sizes=split_shard_sizes,
        dim_concat=dim_concat,
        concat_sizes=concat_shard_sizes,
        group=group,
    )
    expected = torch.split(full, split_shard_sizes, dim=dim_split)[rank].contiguous()

    assert actual.size() == expected.size()
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("shape", "dim_split", "dim_concat"),
    [
        pytest.param((1024, 1024), 0, 1, id="even_split_even_concat_1m"),
        pytest.param((1025, 1024), 0, 1, id="uneven_split_even_concat_1m"),
        pytest.param((1024, 1025), 0, 1, id="even_split_uneven_concat_1m"),
        pytest.param((1025, 1025), 0, 1, id="uneven_split_uneven_concat_1m"),
        pytest.param((64, 129, 128), 1, 2, id="3d_middle_to_last_1m"),
        pytest.param((64, 128, 128), -2, -1, id="negative_dims_equal_even_1m"),
    ],
)
def test_alltoall_transpose_redistributes_between_sharded_layouts(
    shape: tuple[int, ...],
    dim_split: int,
    dim_concat: int,
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    if distributed_backend == "gloo" and torch_version_less_than(2, 6):
        pytest.skip("Gloo alltoall_transpose requires torch >= 2.6.")
    run_distributed_test(
        _test_alltoall_transpose_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=shape,
        dim_split=dim_split,
        dim_concat=dim_concat,
    )


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("split_shard_size_pattern", "concat_shard_size_pattern"),
    [
        pytest.param(
            (128, 384, 256, 512),
            (512, 256, 128, 384),
            id=(
                "split-dim=0,irregular-split-shard-sizes=[128,384,256,512],"
                "concat-dim=1,concat-shard-sizes=[512,256,128,384]"
            ),
        ),
        pytest.param(
            (256, 0, 256, 256),
            (0, 256, 256, 256),
            id=(
                "split-dim=0,empty-split-shard-sizes=[256,0,256,256]," "concat-dim=1,concat-shard-sizes=[0,256,256,256]"
            ),
        ),
    ],
)
def test_alltoall_transpose_supports_explicit_shard_sizes(
    split_shard_size_pattern: tuple[int, ...],
    concat_shard_size_pattern: tuple[int, ...],
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    if distributed_backend == "gloo" and torch_version_less_than(2, 6):
        pytest.skip("Gloo alltoall_transpose requires torch >= 2.6.")
    split_shard_sizes = shard_sizes_from_pattern(split_shard_size_pattern, distributed_world_size)
    concat_shard_sizes = shard_sizes_from_pattern(concat_shard_size_pattern, distributed_world_size)
    run_distributed_test(
        _test_alltoall_transpose_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=(sum(split_shard_sizes), sum(concat_shard_sizes)),
        dim_split=0,
        dim_concat=1,
        split_shard_sizes=split_shard_sizes,
        concat_shard_sizes=concat_shard_sizes,
    )


def _test_invalid_dim_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    primitive: str,
    dim: int,
) -> None:
    full = torch.arange(torch.Size((16, 16)).numel(), dtype=torch.float32, device=device).reshape(16, 16)
    sizes = get_balanced_partition_sizes(full.size(0), world_size)
    local = torch.split(full, sizes, dim=0)[rank].contiguous()

    with pytest.raises(AssertionError):
        if primitive == "split":
            _split(full, dim_=dim, sizes_=sizes, group=group)
        elif primitive == "gather":
            _gather(local, dim_=dim, sizes=sizes, group=group)
        elif primitive == "expand":
            _expand_sharded_tensor(local, dim_=dim, sizes=sizes, group=group)
        else:
            msg = f"Unknown primitive: {primitive}"
            raise ValueError(msg)


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("primitive", "dim"),
    [
        pytest.param("split", 2, id="split_invalid_positive_dim"),
        pytest.param("gather", 2, id="gather_invalid_positive_dim"),
        pytest.param("expand", 2, id="expand_invalid_positive_dim"),
    ],
)
def test_primitives_reject_invalid_dimensions(
    primitive: str, dim: int, distributed_backend: str, distributed_world_size: int
) -> None:
    run_distributed_test(
        _test_invalid_dim_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        primitive=primitive,
        dim=dim,
    )


def _test_expand_rejects_wrong_local_size_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
) -> None:
    sizes = get_balanced_partition_sizes(16, world_size)
    wrong_local_size = sizes[rank] + 1
    local = torch.arange(wrong_local_size * 16, dtype=torch.float32, device=device).reshape(wrong_local_size, 16)

    with pytest.raises(AssertionError):
        _expand_sharded_tensor(local, dim_=0, sizes=sizes, group=group)


@pytest.mark.distributed
def test_expand_sharded_tensor_rejects_wrong_local_size(distributed_backend: str, distributed_world_size: int) -> None:
    run_distributed_test(
        _test_expand_rejects_wrong_local_size_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
    )


def _test_alltoall_transpose_rejects_same_dimension_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
) -> None:
    local = torch.arange(torch.Size((8, 16)).numel(), dtype=torch.float32, device=device).reshape(8, 16)
    sizes = get_balanced_partition_sizes(local.size(1), world_size)

    with pytest.raises(AssertionError):
        _alltoall_transpose(
            local,
            dim_split=1,
            split_sizes=sizes,
            dim_concat=-1,
            concat_sizes=sizes,
            group=group,
        )


@pytest.mark.distributed
def test_alltoall_transpose_rejects_same_split_and_concat_dimension(
    distributed_backend: str, distributed_world_size: int
) -> None:
    run_distributed_test(
        _test_alltoall_transpose_rejects_same_dimension_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
    )


def _test_split_non_contiguous_rank(
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
    full = torch.arange(torch.Size(shape).numel(), dtype=torch.float32, device=device).reshape(shape[1], shape[0]).t()
    assert full.size() == torch.Size(shape)
    assert not full.is_contiguous()

    sizes = get_balanced_partition_sizes(full.size(dim), world_size)
    actual = _split(full, dim_=dim, sizes_=sizes, group=group)
    expected = torch.split(full, sizes, dim=dim)[rank].contiguous()

    assert actual.size() == expected.size()
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


NON_CONTIGUOUS_PARTITION_CASES = [
    pytest.param((1024, 1025), 1, id="transposed_dim1_1m"),
]


@pytest.mark.distributed
@pytest.mark.parametrize(("shape", "dim"), NON_CONTIGUOUS_PARTITION_CASES)
def test_split_accepts_non_contiguous_input(
    shape: tuple[int, ...], dim: int, distributed_backend: str, distributed_world_size: int
) -> None:
    run_distributed_test(
        _test_split_non_contiguous_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=shape,
        dim=dim,
    )


def _test_gather_non_contiguous_rank(
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
    full = torch.arange(torch.Size(shape).numel(), dtype=torch.float32, device=device).reshape(shape[1], shape[0]).t()
    assert full.size() == torch.Size(shape)
    assert not full.is_contiguous()

    sizes = get_balanced_partition_sizes(full.size(dim), world_size)
    local = torch.split(full, sizes, dim=dim)[rank]
    assert not local.is_contiguous()

    actual = _gather(local, dim_=dim, sizes=sizes, group=group)
    expected = full.contiguous()

    assert actual.size() == expected.size()
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.distributed
@pytest.mark.parametrize(("shape", "dim"), NON_CONTIGUOUS_PARTITION_CASES)
def test_gather_accepts_non_contiguous_input(
    shape: tuple[int, ...], dim: int, distributed_backend: str, distributed_world_size: int
) -> None:
    run_distributed_test(
        _test_gather_non_contiguous_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=shape,
        dim=dim,
    )


def _test_reduce_non_contiguous_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    shape: tuple[int, ...],
    use_fp32: bool,
    atol: float = GLOBAL_DEFAULT_ATOL,
    rtol: float = GLOBAL_DEFAULT_RTOL,
) -> None:
    base = torch.arange(torch.Size(shape).numel(), dtype=torch.float32, device=device).reshape(shape[1], shape[0]).t()
    assert base.size() == torch.Size(shape)
    assert not base.is_contiguous()

    local = (base + rank).t().contiguous().t()
    assert not local.is_contiguous()

    actual = _reduce(local, use_fp32=use_fp32, group=group)

    # Build the conceptual per-rank inputs as a leading "rank" dimension:
    # inputs[q] == base + q. Summing over dim=0 gives the all-reduce result.
    rank_offsets = torch.arange(world_size, dtype=torch.float32, device=device).reshape(world_size, *([1] * base.ndim))
    inputs = base.unsqueeze(0) + rank_offsets
    if use_fp32:
        expected = torch.sum(inputs.float(), dim=0).to(local.dtype)
    else:
        expected = torch.sum(inputs, dim=0)

    assert actual.size() == expected.size()
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


NON_CONTIGUOUS_REDUCE_CASES = [
    pytest.param((1024, 1025), True, id="transposed_reduce_fp32_1m"),
    pytest.param((1024, 1025), False, id="transposed_reduce_native_1m"),
]


@pytest.mark.distributed
@pytest.mark.parametrize(("shape", "use_fp32"), NON_CONTIGUOUS_REDUCE_CASES)
def test_reduce_accepts_non_contiguous_input(
    shape: tuple[int, ...], use_fp32: bool, distributed_backend: str, distributed_world_size: int
) -> None:
    run_distributed_test(
        _test_reduce_non_contiguous_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=shape,
        use_fp32=use_fp32,
    )


def _test_split_channels_last_rank(
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
    full = torch.arange(torch.Size(shape).numel(), dtype=torch.float32, device=device).reshape(shape)
    full = full.contiguous(memory_format=torch.channels_last)
    assert full.is_contiguous(memory_format=torch.channels_last)

    sizes = get_balanced_partition_sizes(full.size(dim), world_size)
    actual = _split(full, dim_=dim, sizes_=sizes, group=group)
    expected = torch.split(full, sizes, dim=dim)[rank].contiguous(memory_format=torch.channels_last)

    assert actual.is_contiguous(memory_format=torch.channels_last)
    assert actual.size() == expected.size()
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


CHANNELS_LAST_CASES = [
    pytest.param((64, 4, 64, 64), 0, id="channels_last_dim0_1m"),
]


@pytest.mark.distributed
@pytest.mark.parametrize(("shape", "dim"), CHANNELS_LAST_CASES)
def test_split_preserves_channels_last_memory_format(
    shape: tuple[int, ...], dim: int, distributed_backend: str, distributed_world_size: int
) -> None:
    run_distributed_test(
        _test_split_channels_last_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=shape,
        dim=dim,
    )


def _test_gather_channels_last_rank(
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
    full = torch.arange(torch.Size(shape).numel(), dtype=torch.float32, device=device).reshape(shape)
    full = full.contiguous(memory_format=torch.channels_last)
    assert full.is_contiguous(memory_format=torch.channels_last)

    sizes = get_balanced_partition_sizes(full.size(dim), world_size)
    local = torch.split(full, sizes, dim=dim)[rank].contiguous(memory_format=torch.channels_last)

    actual = _gather(local, dim_=dim, sizes=sizes, group=group)

    assert actual.is_contiguous(memory_format=torch.channels_last)
    assert actual.size() == full.size()
    assert actual.dtype == full.dtype
    assert actual.device == full.device
    torch.testing.assert_close(actual, full, atol=atol, rtol=rtol)


@pytest.mark.distributed
@pytest.mark.parametrize(("shape", "dim"), CHANNELS_LAST_CASES)
def test_gather_preserves_channels_last_memory_format(
    shape: tuple[int, ...], dim: int, distributed_backend: str, distributed_world_size: int
) -> None:
    run_distributed_test(
        _test_gather_channels_last_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=shape,
        dim=dim,
    )


def _test_expand_channels_last_rank(
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
    full = torch.arange(torch.Size(shape).numel(), dtype=torch.float32, device=device).reshape(shape)
    full = full.contiguous(memory_format=torch.channels_last)
    assert full.is_contiguous(memory_format=torch.channels_last)

    sizes = get_balanced_partition_sizes(full.size(dim), world_size)
    local = torch.split(full, sizes, dim=dim)[rank].contiguous(memory_format=torch.channels_last)

    actual = _expand_sharded_tensor(local, dim_=dim, sizes=sizes, group=group)

    assert actual.is_contiguous(memory_format=torch.channels_last)
    assert actual.size() == full.size()
    assert actual.dtype == full.dtype
    assert actual.device == full.device

    normalized_dim = dim % full.dim()
    start = sum(sizes[:rank])
    actual_local_slice = actual.narrow(normalized_dim, start, sizes[rank])

    assert actual_local_slice.size() == local.size()
    assert actual_local_slice.dtype == local.dtype
    assert actual_local_slice.device == local.device
    torch.testing.assert_close(actual_local_slice, local, atol=atol, rtol=rtol)


@pytest.mark.distributed
@pytest.mark.parametrize(("shape", "dim"), CHANNELS_LAST_CASES)
def test_expand_sharded_tensor_preserves_channels_last_memory_format(
    shape: tuple[int, ...], dim: int, distributed_backend: str, distributed_world_size: int
) -> None:
    run_distributed_test(
        _test_expand_channels_last_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=shape,
        dim=dim,
    )


def _test_alltoallwrapper_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    atol: float = GLOBAL_DEFAULT_ATOL,
    rtol: float = GLOBAL_DEFAULT_RTOL,
) -> None:
    input_list = [
        torch.full((128, 128), rank * 10 + dst, dtype=torch.float32, device=device) for dst in range(world_size)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]

    _alltoallwrapper(output_list, input_list, group=group)

    for src, output in enumerate(output_list):
        expected = torch.full_like(output, src * 10 + rank)
        assert output.size() == expected.size()
        assert output.dtype == expected.dtype
        assert output.device == expected.device
        torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)


@pytest.mark.distributed
def test_alltoallwrapper_exchanges_rank_ordered_tensor_lists(
    distributed_backend: str, distributed_world_size: int
) -> None:
    if distributed_backend == "gloo" and torch_version_less_than(2, 6):
        pytest.skip("Gloo alltoallwrapper requires torch >= 2.6.")
    run_distributed_test(
        _test_alltoallwrapper_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
    )


def _test_alltoallwrapper_custom_message_sizes_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    message_sizes: list[int],
    atol: float = GLOBAL_DEFAULT_ATOL,
    rtol: float = GLOBAL_DEFAULT_RTOL,
) -> None:
    input_list = [
        torch.full((message_sizes[dst], message_sizes[rank]), rank * 10 + dst, dtype=torch.float32, device=device)
        for dst in range(world_size)
    ]
    output_list = [
        torch.empty((message_sizes[rank], message_sizes[src]), dtype=torch.float32, device=device)
        for src in range(world_size)
    ]

    _alltoallwrapper(output_list, input_list, group=group)

    for src, output in enumerate(output_list):
        expected = torch.full_like(output, src * 10 + rank)
        torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)


@pytest.mark.distributed
@pytest.mark.parametrize(
    "message_size_pattern",
    [
        pytest.param((128, 384, 256, 512), id="irregular-message-sizes=[128,384,256,512]"),
        pytest.param((256, 0, 256, 256), id="empty-message-sizes=[256,0,256,256]"),
    ],
)
def test_alltoallwrapper_supports_custom_message_sizes(
    message_size_pattern: tuple[int, ...], distributed_backend: str, distributed_world_size: int
) -> None:
    if distributed_backend == "gloo" and torch_version_less_than(2, 6):
        pytest.skip("Gloo alltoallwrapper requires torch >= 2.6.")
    message_sizes = shard_sizes_from_pattern(message_size_pattern, distributed_world_size)
    run_distributed_test(
        _test_alltoallwrapper_custom_message_sizes_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        message_sizes=message_sizes,
    )
