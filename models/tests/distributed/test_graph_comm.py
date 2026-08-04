# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the autograd-aware communication functions in ``anemoi.models.distributed.graph``.

Each tensor communication wrapper covered here is verified against its
contract: the forward layout transformation, the backward gradient rule, and
the ``mgroup=None`` identity fallback. Expected values are computed inline
from plain PyTorch operations. Low-level collective behavior (backend paths,
uneven shards, memory formats) is covered by
``test_communication_primitives.py``. Graph-topology communication such as
``halo_exchange`` belongs with the graph-sharding tests and their ``HaloInfo``
fixtures.

Backward tests build ``loss = (output * grad_output).sum()``, so the output's
incoming gradient is exactly ``grad_output``. It is rank-scaled and
position-varying, so mis-routed slices and missing or spurious cross-rank
reductions produce distinct gradients, and integer-valued, so all expectations
are exact in float32.

Distributed cases are gated behind the ``distributed`` marker and run via::

    pytest models/tests/distributed/test_graph_comm.py \\
        --distributed --distributed-backend={gloo,nccl,all} --distributed-world-size=N
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from distributed_runner import run_distributed_test

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.graph import all_to_all_transpose
from anemoi.models.distributed.graph import ensure_sharded
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import reduce_shard_tensor
from anemoi.models.distributed.graph import reduce_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.graph import sync_tensor

GLOBAL_DEFAULT_ATOL = 1e-12
GLOBAL_DEFAULT_RTOL = 1e-12


def _torch_version_less_than(major: int, minor: int) -> bool:
    """Return True if the installed torch version is older than ``major.minor``."""
    version_parts = torch.__version__.split("+", maxsplit=1)[0].split(".")
    return (int(version_parts[0]), int(version_parts[1])) < (major, minor)


def _make_range_tensor(
    shape: tuple[int, ...],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a float32 tensor of the given shape filled with its flat indices (0, 1, 2, ...)."""
    numel = torch.Size(shape).numel()
    return torch.arange(numel, dtype=torch.float32, device=device).reshape(shape)


def _make_grad_output(
    shape: tuple[int, ...],
    device: torch.device | None = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """Create an integer-valued grad_output: a repeating ``1..251`` pattern times ``scale``."""
    # Small bounded values keep expectations exact in float32; variation exposes mis-routed slices.
    numel = torch.Size(shape).numel()
    values = torch.arange(numel, dtype=torch.float32, device=device) % 251 + 1
    return (values * scale).reshape(shape)


def test_ensure_sharded_computes_sizes_without_group() -> None:
    """Without a group, compute single-rank sizes and return the tensor unchanged."""
    x = _make_range_tensor((4, 3))
    expected = x.clone()

    sharded, sizes = ensure_sharded(x, dim=0, shard_sizes=None, model_comm_group=None)

    assert sizes == [4]
    torch.testing.assert_close(sharded, expected)


def test_ensure_sharded_validates_given_sizes() -> None:
    """Consistent given sizes pass the tensor through; inconsistent sizes raise."""
    x = _make_range_tensor((4, 3))

    sharded, sizes = ensure_sharded(x, dim=0, shard_sizes=[4], model_comm_group=None)

    assert sharded is x
    assert sizes == [4]

    with pytest.raises(AssertionError, match="expected shard size"):
        ensure_sharded(x, dim=0, shard_sizes=[5], model_comm_group=None)


def _test_ensure_sharded_rank(
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
    full = _make_range_tensor(shape, device)
    expected_sizes = get_balanced_partition_sizes(full.size(dim), world_size)
    expected = torch.split(full, expected_sizes, dim=dim)[rank].clone()

    sharded, sizes = ensure_sharded(full, dim=dim, shard_sizes=None, model_comm_group=group)

    assert sizes == expected_sizes
    torch.testing.assert_close(sharded, expected, atol=atol, rtol=rtol)

    validated, validated_sizes = ensure_sharded(
        sharded,
        dim=dim,
        shard_sizes=sizes,
        model_comm_group=group,
    )

    assert validated is sharded
    assert validated_sizes == sizes


@pytest.mark.distributed
def test_ensure_sharded_shards_replicated_tensor(
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    """Shard a replicated tensor with balanced sizes; pass an existing shard through.

    With ``shard_sizes=None`` the tensor is sharded and the computed sizes are
    returned; with matching sizes the shard is returned as the same object
    after a consistency check.
    """
    run_distributed_test(
        _test_ensure_sharded_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=(1025, 1024),
        dim=0,
    )


@pytest.mark.parametrize(
    "gather_in_backward",
    [
        pytest.param(True, id="gather_in_backward"),
        pytest.param(False, id="no_gather_in_backward"),
    ],
)
def test_shard_tensor_identity_without_group(gather_in_backward: bool) -> None:
    """With ``mgroup=None``, values and gradients pass through unchanged.

    Covers the local fallback of ``_ShardParallelSection`` in both backward
    modes: no forward split and no backward communication occur.
    """
    x = _make_range_tensor((4, 3))
    expected = x.clone()
    grad_output = _make_grad_output((4, 3))
    x.requires_grad_(True)

    sharded = shard_tensor(
        x,
        dim=0,
        sizes=None,
        mgroup=None,
        gather_in_backward=gather_in_backward,
    )

    assert sharded.size() == expected.size()
    assert sharded.dtype == expected.dtype
    assert sharded.device == expected.device
    torch.testing.assert_close(sharded, expected)

    loss = (sharded * grad_output).sum()  # d(loss)/d(sharded) == grad_output
    loss.backward()
    torch.testing.assert_close(x.grad, grad_output)


def _test_shard_tensor_rank(
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
    full = _make_range_tensor(shape, device)
    sizes = get_balanced_partition_sizes(full.size(dim), world_size)
    expected = torch.split(full, sizes, dim=dim)[rank].clone()
    full.requires_grad_(True)

    sharded = shard_tensor(full, dim=dim, sizes=sizes, mgroup=group)

    assert sharded.size() == expected.size()
    assert sharded.dtype == expected.dtype
    assert sharded.device == expected.device
    torch.testing.assert_close(sharded, expected, atol=atol, rtol=rtol)

    # each rank's grad_output is its slice of one full grad_output
    grad_output_full = _make_grad_output(shape, device)
    grad_output_local = torch.split(grad_output_full, sizes, dim=dim)[rank].contiguous()
    loss = (sharded * grad_output_local).sum()  # d(loss)/d(sharded) == grad_output_local
    loss.backward()

    torch.testing.assert_close(full.grad, grad_output_full, atol=atol, rtol=rtol)


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("shape", "dim"),
    [
        pytest.param((1025, 1024), 0, id="dim0_uneven_1m"),
        pytest.param((64, 128, 129), -1, id="negative_last_dim_3d_1m"),
    ],
)
def test_shard_tensor_gathers_gradients(
    shape: tuple[int, ...],
    dim: int,
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    """Keep the rank slice forward; all-gather the shard grad_outputs backward.

    Each rank's grad_output is its slice of one conceptual full grad_output,
    so the gathered gradient (default ``gather_in_backward=True``) must
    reassemble that tensor exactly on every rank; reordering or dropping a
    rank's contribution would fail.
    """
    run_distributed_test(
        _test_shard_tensor_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=shape,
        dim=dim,
    )


def _test_shard_tensor_no_backward_gather_rank(
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
    full = _make_range_tensor(shape, device)
    sizes = get_balanced_partition_sizes(full.size(dim), world_size)
    expected = torch.split(full, sizes, dim=dim)[rank].clone()
    full.requires_grad_(True)

    sharded = shard_tensor(full, dim=dim, sizes=sizes, mgroup=group, gather_in_backward=False)

    assert sharded.size() == expected.size()
    torch.testing.assert_close(sharded, expected, atol=atol, rtol=rtol)

    # each rank's grad_output is its slice of one full grad_output
    grad_output_full = _make_grad_output(shape, device)
    grad_output_local = torch.split(grad_output_full, sizes, dim=dim)[rank].contiguous()
    loss = (sharded * grad_output_local).sum()  # d(loss)/d(sharded) == grad_output_local

    with patch(
        "anemoi.models.distributed.graph._gather",
        side_effect=AssertionError("backward must not gather"),
    ):
        loss.backward()

    assert full.grad.size() == full.size()
    assert full.grad.dtype == full.dtype
    assert full.grad.device == full.device

    normalized_dim = dim % full.dim()
    start = sum(sizes[:rank])
    actual_local_slice = full.grad.narrow(normalized_dim, start, sizes[rank])
    torch.testing.assert_close(actual_local_slice, grad_output_local, atol=atol, rtol=rtol)


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("shape", "dim"),
    [
        pytest.param((1025, 1024), 0, id="dim0_uneven_1m"),
    ],
)
def test_shard_tensor_no_backward_gather(
    shape: tuple[int, ...],
    dim: int,
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    """With ``gather_in_backward=False``, only the rank-local gradient slice is filled.

    Forward is unchanged. Backward performs no communication
    (``_expand_sharded_tensor``): the remaining gradient slices are
    uninitialized by design and must never be read.
    """
    run_distributed_test(
        _test_shard_tensor_no_backward_gather_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=shape,
        dim=dim,
    )

