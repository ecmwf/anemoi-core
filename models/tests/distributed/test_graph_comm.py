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


@pytest.mark.parametrize(
    "sizes",
    [
        pytest.param(None, id="sizes_none"),
        pytest.param([4], id="explicit_single_rank_sizes"),
    ],
)
def test_gather_tensor_identity_without_group(sizes: list[int] | None) -> None:
    """With ``mgroup=None``, values and gradients pass through unchanged.

    Covers the local fallback of ``_GatherParallelSection``; the shard-size
    metadata is ignored on this path.
    """
    x = _make_range_tensor((4, 3))
    expected = x.clone()
    grad_output = _make_grad_output((4, 3))
    x.requires_grad_(True)

    gathered = gather_tensor(x, dim=0, sizes=sizes, mgroup=None)

    assert gathered.size() == expected.size()
    assert gathered.dtype == expected.dtype
    assert gathered.device == expected.device
    torch.testing.assert_close(gathered, expected)

    loss = (gathered * grad_output).sum()  # d(loss)/d(gathered) == grad_output
    loss.backward()
    torch.testing.assert_close(x.grad, grad_output)


def _test_gather_tensor_rank(
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
    local = torch.split(full, sizes, dim=dim)[rank].contiguous().clone().requires_grad_(True)

    gathered = gather_tensor(local, dim=dim, sizes=sizes, mgroup=group)

    assert gathered.size() == full.size()
    assert gathered.dtype == full.dtype
    assert gathered.device == full.device
    torch.testing.assert_close(gathered, full, atol=atol, rtol=rtol)

    grad_output = _make_grad_output(shape, device, scale=rank + 1)
    loss = (gathered * grad_output).sum()  # d(loss)/d(gathered) == grad_output
    loss.backward()

    expected_grad = torch.split(grad_output, sizes, dim=dim)[rank].clone()
    torch.testing.assert_close(local.grad, expected_grad, atol=atol, rtol=rtol)


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("shape", "dim"),
    [
        pytest.param((1025, 1024), 0, id="dim0_uneven_1m"),
        pytest.param((1024, 1025), 1, id="dim1_uneven_1m"),
        pytest.param((64, 128, 129), -1, id="negative_last_dim_3d_1m"),
    ],
)
def test_gather_tensor_splits_gradients(
    shape: tuple[int, ...],
    dim: int,
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    """Gather shards forward; split each rank's own grad_output backward.

    Every rank reconstructs the full tensor forward. Backward involves no
    communication: with rank-scaled grad_outputs the expected gradient is the
    rank's slice of its own grad_output, so a spurious cross-rank all-reduce
    would produce the slice of the rank-summed grad_outputs and fail.
    """
    run_distributed_test(
        _test_gather_tensor_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=shape,
        dim=dim,
    )


def test_reduce_tensor_identity_without_group() -> None:
    """With ``mgroup=None``, values and gradients pass through unchanged.

    Covers the local fallback of ``_ReduceParallelSection``.
    """
    x = _make_range_tensor((4, 3))
    expected = x.clone()
    grad_output = _make_grad_output((4, 3))
    x.requires_grad_(True)

    reduced = reduce_tensor(x, mgroup=None)

    assert reduced.size() == expected.size()
    assert reduced.dtype == expected.dtype
    assert reduced.device == expected.device
    torch.testing.assert_close(reduced, expected)

    loss = (reduced * grad_output).sum()  # d(loss)/d(reduced) == grad_output
    loss.backward()
    torch.testing.assert_close(x.grad, grad_output)


def _test_reduce_tensor_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    shape: tuple[int, ...],
    atol: float = GLOBAL_DEFAULT_ATOL,
    rtol: float = GLOBAL_DEFAULT_RTOL,
) -> None:
    base = _make_range_tensor(shape, device)
    local = (base + rank).clone().requires_grad_(True)

    reduced = reduce_tensor(local, mgroup=group)

    expected = torch.stack([base + q for q in range(world_size)]).sum(dim=0)
    assert reduced.size() == expected.size()
    assert reduced.dtype == expected.dtype
    assert reduced.device == expected.device
    torch.testing.assert_close(reduced, expected, atol=atol, rtol=rtol)

    grad_output = _make_grad_output(shape, device, scale=rank + 1)
    loss = (reduced * grad_output).sum()  # d(loss)/d(reduced) == grad_output
    loss.backward()

    torch.testing.assert_close(local.grad, grad_output, atol=atol, rtol=rtol)


@pytest.mark.distributed
def test_reduce_tensor_backward_is_identity(
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    """Sum same-shaped rank contributions forward; keep grad_output unchanged backward.

    The identity backward is deliberately not the adjoint of the forward sum.
    With rank-scaled grad_outputs, a spurious backward all-reduce would
    return the rank-summed grad_outputs instead of the rank's own and fail.
    """
    run_distributed_test(
        _test_reduce_tensor_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=(1025, 1024),
    )


@pytest.mark.parametrize(
    "gather_in_fwd",
    [
        pytest.param(True, id="gather_in_fwd"),
        pytest.param(False, id="no_gather_in_fwd"),
    ],
)
def test_sync_tensor_identity_without_group(gather_in_fwd: bool) -> None:
    """With ``mgroup=None``, values and gradients pass through unchanged.

    Covers the local fallback of ``_SyncParallelSection`` in both forward
    modes: neither a gather nor a backward reduction occurs.
    """
    x = _make_range_tensor((4, 3))
    expected = x.clone()
    grad_output = _make_grad_output((4, 3))
    x.requires_grad_(True)

    synced = sync_tensor(x, dim=0, sizes=None, mgroup=None, gather_in_fwd=gather_in_fwd)

    assert synced.size() == expected.size()
    assert synced.dtype == expected.dtype
    assert synced.device == expected.device
    torch.testing.assert_close(synced, expected)

    loss = (synced * grad_output).sum()  # d(loss)/d(synced) == grad_output
    loss.backward()
    torch.testing.assert_close(x.grad, grad_output)


def _test_sync_tensor_rank(
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
    local = torch.split(full, sizes, dim=dim)[rank].contiguous().clone()
    local.requires_grad_(True)

    synced = sync_tensor(local, dim=dim, sizes=sizes, mgroup=group)

    assert synced.size() == full.size()
    assert synced.dtype == full.dtype
    assert synced.device == full.device
    torch.testing.assert_close(synced, full, atol=atol, rtol=rtol)

    grad_output = _make_grad_output(shape, device, scale=rank + 1)
    loss = (synced * grad_output).sum()  # d(loss)/d(synced) == grad_output
    loss.backward()

    # the backward all-reduce sums the rank scales: sum_q (q+1) = ws(ws+1)/2
    grad_output_sum = _make_grad_output(shape, device, scale=world_size * (world_size + 1) / 2)
    expected_grad = torch.split(grad_output_sum, sizes, dim=dim)[rank].clone()
    torch.testing.assert_close(local.grad, expected_grad, atol=atol, rtol=rtol)


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("shape", "dim"),
    [
        pytest.param((1025, 1024), 0, id="dim0_uneven_1m"),
        pytest.param((64, 128, 129), -1, id="negative_last_dim_3d_1m"),
    ],
)
def test_sync_tensor_gathers_and_reduces_gradients(
    shape: tuple[int, ...],
    dim: int,
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    """Gather shards forward; all-reduce the grad_outputs, then split backward.

    The forward matches gather_tensor. The backward is the exact adjoint of
    gather-and-replicate: the rank-scaled grad_outputs are summed across
    ranks, then each rank keeps its slice. This is the observable difference
    from gather_tensor, whose backward keeps the rank's own grad_output
    slice without communication.
    """
    run_distributed_test(
        _test_sync_tensor_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=shape,
        dim=dim,
    )


def _test_sync_tensor_no_gather_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    shape: tuple[int, ...],
    gather_in_fwd: bool,
    with_sizes: bool,
    atol: float = GLOBAL_DEFAULT_ATOL,
    rtol: float = GLOBAL_DEFAULT_RTOL,
) -> None:
    base = _make_range_tensor(shape, device)
    local = base + rank
    expected = local.clone()
    local.requires_grad_(True)

    sizes = get_balanced_partition_sizes(shape[0], world_size) if with_sizes else None
    synced = sync_tensor(local, dim=0, sizes=sizes, mgroup=group, gather_in_fwd=gather_in_fwd)

    assert synced.size() == expected.size()
    assert synced.dtype == expected.dtype
    assert synced.device == expected.device
    torch.testing.assert_close(synced, expected, atol=atol, rtol=rtol)

    grad_output = _make_grad_output(shape, device, scale=rank + 1)
    loss = (synced * grad_output).sum()  # d(loss)/d(synced) == grad_output
    loss.backward()

    # the backward all-reduce sums the rank scales: sum_q (q+1) = ws(ws+1)/2
    expected_grad = _make_grad_output(shape, device, scale=world_size * (world_size + 1) / 2)
    torch.testing.assert_close(local.grad, expected_grad, atol=atol, rtol=rtol)


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("gather_in_fwd", "with_sizes"),
    [
        pytest.param(False, True, id="gather_in_fwd_false"),
        pytest.param(True, False, id="sizes_none"),
    ],
)
def test_sync_tensor_no_forward_gather(
    gather_in_fwd: bool,
    with_sizes: bool,
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    """Without a forward gather, the forward is the identity; the backward still all-reduces.

    The forward gathers only when ``gather_in_fwd`` is true and sizes are
    given; both disabling branches are covered. Having not gathered, the
    backward returns the reduced grad_output whole, without splitting.
    Inputs must be same-shaped across ranks.
    """
    run_distributed_test(
        _test_sync_tensor_no_gather_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=(1025, 1024),
        gather_in_fwd=gather_in_fwd,
        with_sizes=with_sizes,
    )


def test_reduce_shard_tensor_identity_without_group() -> None:
    """With ``mgroup=None``, values and gradients pass through unchanged.

    Covers the local fallback of ``_ReduceShardParallelSection``.
    """
    x = _make_range_tensor((4, 3))
    expected = x.clone()
    grad_output = _make_grad_output((4, 3))
    x.requires_grad_(True)

    shard_sum = reduce_shard_tensor(x, dim=0, sizes=None, mgroup=None)

    assert shard_sum.size() == expected.size()
    assert shard_sum.dtype == expected.dtype
    assert shard_sum.device == expected.device
    torch.testing.assert_close(shard_sum, expected)

    loss = (shard_sum * grad_output).sum()  # d(loss)/d(shard_sum) == grad_output
    loss.backward()
    torch.testing.assert_close(x.grad, grad_output)


def _test_reduce_shard_tensor_rank(
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
    base = _make_range_tensor(shape, device)
    sizes = get_balanced_partition_sizes(base.size(dim), world_size)
    expected_full = torch.stack([base + q for q in range(world_size)]).sum(dim=0)
    expected = torch.split(expected_full, sizes, dim=dim)[rank].clone()

    local = base + rank
    local.requires_grad_(True)

    shard_sum = reduce_shard_tensor(local, dim=dim, sizes=sizes, mgroup=group)

    assert shard_sum.size() == expected.size()
    assert shard_sum.dtype == expected.dtype
    assert shard_sum.device == expected.device
    torch.testing.assert_close(shard_sum, expected, atol=atol, rtol=rtol)

    # each rank's grad_output is its slice of one full grad_output
    grad_output_full = _make_grad_output(shape, device)
    grad_output_local = torch.split(grad_output_full, sizes, dim=dim)[rank].contiguous()
    loss = (shard_sum * grad_output_local).sum()  # d(loss)/d(shard_sum) == grad_output_local
    loss.backward()

    torch.testing.assert_close(local.grad, grad_output_full, atol=atol, rtol=rtol)


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("shape", "dim"),
    [
        pytest.param((1025, 1024), 0, id="dim0_uneven_1m"),
        pytest.param((64, 128, 129), -1, id="negative_last_dim_3d_1m"),
    ],
)
def test_reduce_shard_tensor_gathers_gradients(
    shape: tuple[int, ...],
    dim: int,
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    """All-reduce full-shape contributions and keep the rank slice forward; gather backward.

    The forward is semantically a reduce-scatter. The backward is the exact
    adjoint: the per-rank shard grad_outputs are all-gathered into a full
    gradient on every rank. Each rank's grad_output is its slice of one
    conceptual full grad_output, so the gathered gradient must reassemble
    that tensor exactly.
    """
    run_distributed_test(
        _test_reduce_shard_tensor_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=shape,
        dim=dim,
    )


def test_all_to_all_transpose_identity_without_group() -> None:
    """With ``mgroup=None``, values and gradients pass through unchanged.

    Covers the local fallback of ``_AllToAllParallelSection``; the layout
    metadata is ignored on this path.
    """
    x = _make_range_tensor((4, 3))
    expected = x.clone()
    grad_output = _make_grad_output((4, 3))
    x.requires_grad_(True)

    transposed = all_to_all_transpose(
        x,
        dim_split=0,
        split_sizes=None,
        dim_concat=1,
        concat_sizes=None,
        mgroup=None,
    )

    assert transposed.size() == expected.size()
    assert transposed.dtype == expected.dtype
    assert transposed.device == expected.device
    torch.testing.assert_close(transposed, expected)

    loss = (transposed * grad_output).sum()  # d(loss)/d(transposed) == grad_output
    loss.backward()
    torch.testing.assert_close(x.grad, grad_output)


def _test_all_to_all_transpose_rank(
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
    full = _make_range_tensor(shape, device)
    split_sizes = get_balanced_partition_sizes(full.size(dim_split), world_size)
    concat_sizes = get_balanced_partition_sizes(full.size(dim_concat), world_size)
    expected = torch.split(full, split_sizes, dim=dim_split)[rank].clone()

    local = torch.split(full, concat_sizes, dim=dim_concat)[rank].contiguous().clone()
    local.requires_grad_(True)

    transposed = all_to_all_transpose(
        local,
        dim_split=dim_split,
        split_sizes=split_sizes,
        dim_concat=dim_concat,
        concat_sizes=concat_sizes,
        mgroup=group,
    )

    assert transposed.size() == expected.size()
    assert transposed.dtype == expected.dtype
    assert transposed.device == expected.device
    torch.testing.assert_close(transposed, expected, atol=atol, rtol=rtol)

    # each rank's grad_output is its split-layout slice of one full grad_output
    grad_output_full = _make_grad_output(shape, device)
    grad_output = torch.split(grad_output_full, split_sizes, dim=dim_split)[rank].contiguous()
    loss = (transposed * grad_output).sum()  # d(loss)/d(transposed) == grad_output
    loss.backward()

    # the reverse transpose delivers the rank's concat-layout slice
    expected_grad = torch.split(grad_output_full, concat_sizes, dim=dim_concat)[rank].clone()
    torch.testing.assert_close(local.grad, expected_grad, atol=atol, rtol=rtol)


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("shape", "dim_split", "dim_concat"),
    [
        pytest.param((1025, 1025), 0, 1, id="uneven_both_dims_1m"),
        pytest.param((64, 128, 129), 1, -1, id="negative_concat_dim_3d_1m"),
    ],
)
def test_all_to_all_transpose_inverts_gradients(
    shape: tuple[int, ...],
    dim_split: int,
    dim_concat: int,
    distributed_backend: str,
    distributed_world_size: int,
) -> None:
    """Redistribute between sharded layouts forward; apply the inverse backward.

    Each rank starts with its concat-layout slice of a conceptual tensor and
    receives its split-layout slice, never materializing the full tensor.
    The backward runs the reverse redistribution: with each rank's
    grad_output being its split-layout slice of one full grad_output, the
    input gradient must be the rank's concat-layout slice of that tensor.
    """
    if distributed_backend == "gloo" and _torch_version_less_than(2, 6):
        pytest.skip("Gloo all_to_all_transpose requires torch >= 2.6.")
    run_distributed_test(
        _test_all_to_all_transpose_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        shape=shape,
        dim_split=dim_split,
        dim_concat=dim_concat,
    )
