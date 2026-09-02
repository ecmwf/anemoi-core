# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Distributed tests for communication interfaces

Tests the distributed communication interfaces compile and the output matches the eager output.

These tests are skipped by default. Pass ``--distributed`` to run them. Use
``--distributed-backend`` and ``--distributed-world-size`` to select the backend
and rank count.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.primitives import _alltoall_op
from anemoi.models.distributed.primitives import _alltoall_transpose
from anemoi.models.distributed.primitives import _resolve_group_name
from anemoi.models.testing import run_distributed_test

GLOBAL_DEFAULT_ATOL = 1e-12
GLOBAL_DEFAULT_RTOL = 1e-12


def _torch_version_less_than(major: int, minor: int) -> bool:
    # TODO: Move to shared util or remove.
    version_parts = torch.__version__.split("+", maxsplit=1)[0].split(".")
    return (int(version_parts[0]), int(version_parts[1])) < (major, minor)


def _test_alltoall_opcheck_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    shape: tuple[int, ...],
    dim_split: int,
    dim_concat: int,
) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    full = torch.arange(torch.Size(shape).numel(), dtype=torch.float32, device=device).reshape(shape)
    split_sizes = get_balanced_partition_sizes(full.size(dim_split), world_size)
    concat_sizes = get_balanced_partition_sizes(full.size(dim_concat), world_size)
    local = torch.split(full, concat_sizes, dim=dim_concat)[rank].contiguous()

    # The custom op operates on the already-split input list; split/concat live
    # outside the op in _alltoall_transpose and stay traceable by torch.compile.
    ndim = local.dim()
    norm_split = dim_split % ndim
    norm_concat = dim_concat % ndim
    input_list = [x.contiguous() for x in torch.split(local, split_sizes, dim=norm_split)]

    # shape received from each rank: dim_split = this rank's split size,
    # dim_concat = that rank's concat size. Flattened into a single list[int]
    # because custom-op schemas don't support list[list[int]].
    output_shapes_flat: list[int] = []
    for src in range(world_size):
        out_shape = list(local.shape)
        out_shape[norm_split] = split_sizes[rank]
        out_shape[norm_concat] = concat_sizes[src]
        output_shapes_flat.extend(out_shape)

    group_name = _resolve_group_name(group)
    args = (input_list, output_shapes_flat, ndim, group_name)

    # Validate the custom op registration: schema, fake/meta implementation and
    # (where applicable) autograd registration are all exercised by opcheck.
    torch.library.opcheck(_alltoall_op, args)

    # Explicitly verify the registered fake implementation returns the same shapes,
    # dtypes and devices as the real (eager) implementation for every output tensor.
    real = _alltoall_op(*args)
    with FakeTensorMode(allow_non_fake_inputs=True) as fake_mode:
        fake_input_list = [fake_mode.from_tensor(t) for t in input_list]
        fake = _alltoall_op(fake_input_list, output_shapes_flat, ndim, group_name)

    assert len(fake) == len(real), f"fake returned {len(fake)} tensors, real returned {len(real)}"
    for i, (fake_t, real_t) in enumerate(zip(fake, real)):
        assert fake_t.shape == real_t.shape, f"output {i}: fake shape {fake_t.shape} != real shape {real_t.shape}"
        assert fake_t.dtype == real_t.dtype, f"output {i}: fake dtype {fake_t.dtype} != real dtype {real_t.dtype}"
        assert fake_t.device == real_t.device, f"output {i}: fake device {fake_t.device} != real device {real_t.device}"


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("shape", "dim_split", "dim_concat"),
    [
        pytest.param((1024, 1024), 0, 1, id="even_split_even_concat_1m"),
        pytest.param((1025, 1024), 0, 1, id="uneven_split_even_concat_1m"),
        pytest.param((1024, 1025), 0, 1, id="even_split_uneven_concat_1m"),
        pytest.param((1025, 1025), 0, 1, id="uneven_split_uneven_concat_1m"),
        pytest.param((64, 129, 128), 1, 2, id="3d_middle_to_last_1m"),
        pytest.param((64, 128, 129), 1, -1, id="negative_concat_dim_1m"),
    ],
)
def test_alltoall_op_interface(
    shape: tuple[int, ...],
    dim_split: int,
    dim_concat: int,
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    """Uses torch.library.opcheck to validate the registered custom op interface for the alltoall exchange
    and checks the fake shapes match the real shapes produced.
    """
    if distributed_backend == "gloo" and _torch_version_less_than(2, 6):
        pytest.skip("Gloo alltoall requires torch >= 2.6.")
    run_distributed_test(
        _test_alltoall_opcheck_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=shape,
        dim_split=dim_split,
        dim_concat=dim_concat,
    )


def _test_alltoall_transpose_compile_matches_eager_rank(
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
    full = torch.arange(torch.Size(shape).numel(), dtype=torch.float32, device=device).reshape(shape)
    split_sizes = get_balanced_partition_sizes(full.size(dim_split), world_size)
    concat_sizes = get_balanced_partition_sizes(full.size(dim_concat), world_size)
    local = torch.split(full, concat_sizes, dim=dim_concat)[rank].contiguous()

    eager = _alltoall_transpose(
        local,
        dim_split=dim_split,
        split_sizes=split_sizes,
        dim_concat=dim_concat,
        concat_sizes=concat_sizes,
        group=group,
    )

    # Wrapping the collective in the _alltoall_op custom op should let the surrounding
    # split/concat compile into a single full graph. fullgraph=True raises (rather than
    # silently graph-breaking) if the function cannot be captured, so a compile failure
    # surfaces here as an explicit error.
    torch._dynamo.reset()
    try:
        compiled_transpose = torch.compile(
            _alltoall_transpose,
            fullgraph=True,
            dynamic=False,
            mode="max-autotune",
        )
        compiled = compiled_transpose(
            local,
            dim_split=dim_split,
            split_sizes=split_sizes,
            dim_concat=dim_concat,
            concat_sizes=concat_sizes,
            group=group,
        )
    except Exception as err:
        msg = f"_alltoall_transpose failed to compile with fullgraph=True: {err}"
        raise AssertionError(msg) from err

    assert compiled.size() == eager.size()
    assert compiled.dtype == eager.dtype
    assert compiled.device == eager.device
    torch.testing.assert_close(compiled, eager, atol=atol, rtol=rtol)


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("shape", "dim_split", "dim_concat"),
    [
        pytest.param((1024, 1024), 0, 1, id="even_split_even_concat_1m"),
        pytest.param((1025, 1024), 0, 1, id="uneven_split_even_concat_1m"),
        pytest.param((1024, 1025), 0, 1, id="even_split_uneven_concat_1m"),
        pytest.param((1025, 1025), 0, 1, id="uneven_split_uneven_concat_1m"),
        pytest.param((64, 129, 128), 1, 2, id="3d_middle_to_last_1m"),
        pytest.param((64, 128, 129), 1, -1, id="negative_concat_dim_1m"),
    ],
)
def test_alltoall_transpose_compiles_and_matches_eager(
    shape: tuple[int, ...],
    dim_split: int,
    dim_concat: int,
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    """Compile _alltoall_transpose (which wraps the all-to-all collective as a custom op)
    with fullgraph=True, dynamic=False, mode='max-autotune' and check it both compiles
    and produces the same result as the eager implementation.
    """
    if distributed_backend == "gloo" and _torch_version_less_than(2, 6):
        pytest.skip("Gloo alltoall requires torch >= 2.6.")
    run_distributed_test(
        _test_alltoall_transpose_compile_matches_eager_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=shape,
        dim_split=dim_split,
        dim_concat=dim_concat,
    )
