# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import argparse
import os
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.graph import gather_channels
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import reduce_shard_tensor
from anemoi.models.distributed.graph import reduce_tensor
from anemoi.models.distributed.graph import shard_channels
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.graph import sync_tensor
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.distributed.transformer import shard_heads
from anemoi.models.distributed.transformer import shard_sequence

GLOBAL_DEFAULT_ATOL = 1e-12
GLOBAL_DEFAULT_RTOL = 1e-12


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


def _float32_matmul_precision(default: str = "highest") -> str:
    raw = os.getenv("ANEMOI_DISTRIBUTED_TEST_PRECISION", default)
    value = raw.strip().lower()
    if value not in {"highest", "high", "medium"}:
        msg = "Invalid value for ANEMOI_DISTRIBUTED_TEST_PRECISION: " f"{raw!r}. Use one of: highest, high, medium."
        raise ValueError(msg)
    return value


def _env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None:
        return None

    try:
        value = float(raw)
    except ValueError as exc:
        msg = f"Invalid value for {name}: {raw!r}. Use a non-negative float."
        raise ValueError(msg) from exc

    if value < 0.0:
        msg = f"Invalid value for {name}: {raw!r}. Use a non-negative float."
        raise ValueError(msg)

    return value


def _tolerances() -> tuple[float, float]:
    atol = _env_float("ANEMOI_DISTRIBUTED_TEST_ATOL")
    rtol = _env_float("ANEMOI_DISTRIBUTED_TEST_RTOL")
    return (
        GLOBAL_DEFAULT_ATOL if atol is None else atol,
        GLOBAL_DEFAULT_RTOL if rtol is None else rtol,
    )


def _configure_numeric_settings(*, deterministic: bool, matmul_precision: str) -> None:
    torch.set_float32_matmul_precision(matmul_precision)

    if not deterministic:
        return

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _split_by_shapes(tensor: torch.Tensor, dim: int, shard_shapes: list[list[int]], rank: int) -> torch.Tensor:
    split_sizes = [shape[dim] for shape in shard_shapes]
    return torch.split(tensor, split_sizes, dim=dim)[rank].contiguous()


def _run_core_primitives(
    rank: int,
    world_size: int,
    device: torch.device,
    model_comm_group: dist.ProcessGroup,
) -> None:
    atol, rtol = _tolerances()

    # Keep split sizes equal to ensure backend portability (Gloo all_gather requires equal tensor sizes).
    x_full = torch.arange(24, dtype=torch.float32, device=device).reshape(8, 3)
    shard_shapes_dim0 = get_shard_shapes(x_full, dim=0, model_comm_group=model_comm_group)

    # shard_tensor: forward split + backward gather
    x = x_full.clone().requires_grad_(True)
    x_local_expected = _split_by_shapes(x_full, dim=0, shard_shapes=shard_shapes_dim0, rank=rank)
    x_local = shard_tensor(
        x,
        dim=0,
        shapes=shard_shapes_dim0,
        mgroup=model_comm_group,
        gather_in_backward=True,
    )
    torch.testing.assert_close(x_local, x_local_expected, atol=atol, rtol=rtol)
    x_local.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x), atol=atol, rtol=rtol)

    # gather_tensor: forward gather + backward split
    y_full = torch.arange(42, dtype=torch.float32, device=device).reshape(7, 6)
    shard_shapes_dim1 = get_shard_shapes(y_full, dim=1, model_comm_group=model_comm_group)
    y_local = _split_by_shapes(y_full, dim=1, shard_shapes=shard_shapes_dim1, rank=rank).clone().requires_grad_(True)
    y_gathered = gather_tensor(y_local, dim=1, shapes=shard_shapes_dim1, mgroup=model_comm_group)
    torch.testing.assert_close(y_gathered, y_full, atol=atol, rtol=rtol)
    y_gathered.sum().backward()
    torch.testing.assert_close(y_local.grad, torch.ones_like(y_local), atol=atol, rtol=rtol)

    # reduce_tensor: all-reduce in forward, identity in backward
    z = torch.full((3, 4), float(rank + 1), dtype=torch.float32, device=device, requires_grad=True)
    z_reduced = reduce_tensor(z, model_comm_group)
    expected_sum = float(sum(range(1, world_size + 1)))
    torch.testing.assert_close(
        z_reduced,
        torch.full_like(z, expected_sum),
        atol=atol,
        rtol=rtol,
    )
    z_reduced.sum().backward()
    torch.testing.assert_close(z.grad, torch.ones_like(z), atol=atol, rtol=rtol)

    # sync_tensor gather_in_fwd=True: gather in fwd, reduce+split in bwd
    sync_full = torch.arange(24, dtype=torch.float32, device=device).reshape(8, 3)
    sync_shapes = get_shard_shapes(sync_full, dim=0, model_comm_group=model_comm_group)
    sync_local = _split_by_shapes(sync_full, dim=0, shard_shapes=sync_shapes, rank=rank).clone().requires_grad_(True)
    sync_gathered = sync_tensor(
        sync_local,
        dim=0,
        shapes=sync_shapes,
        mgroup=model_comm_group,
        gather_in_fwd=True,
    )
    torch.testing.assert_close(sync_gathered, sync_full, atol=atol, rtol=rtol)
    sync_gathered.sum().backward()
    torch.testing.assert_close(
        sync_local.grad,
        torch.full_like(sync_local, float(world_size)),
        atol=atol,
        rtol=rtol,
    )

    # sync_tensor gather_in_fwd=False: identity in fwd, all-reduce in bwd.
    # expects same tensor shape across ranks in backward all-reduce.
    sync_replicated = sync_full.clone().requires_grad_(True)
    sync_same = sync_tensor(
        sync_replicated,
        dim=0,
        shapes=sync_shapes,
        mgroup=model_comm_group,
        gather_in_fwd=False,
    )
    torch.testing.assert_close(sync_same, sync_replicated.detach(), atol=atol, rtol=rtol)
    sync_same.sum().backward()
    torch.testing.assert_close(
        sync_replicated.grad,
        torch.full_like(sync_replicated, float(world_size)),
        atol=atol,
        rtol=rtol,
    )

    # reduce_shard_tensor: all-reduce then split in fwd, gather in bwd
    redshard_input = (x_full + float(rank)).clone().requires_grad_(True)
    redshard_out = reduce_shard_tensor(redshard_input, dim=0, shapes=shard_shapes_dim0, mgroup=model_comm_group)
    expected_reduced_full = x_full * world_size + (world_size * (world_size - 1) / 2.0)
    expected_reduced_local = _split_by_shapes(expected_reduced_full, dim=0, shard_shapes=shard_shapes_dim0, rank=rank)
    torch.testing.assert_close(
        redshard_out,
        expected_reduced_local,
        atol=atol,
        rtol=rtol,
    )
    redshard_out.sum().backward()
    torch.testing.assert_close(
        redshard_input.grad,
        torch.ones_like(redshard_input),
        atol=atol,
        rtol=rtol,
    )


def _run_transformer_primitives(
    rank: int,
    world_size: int,
    device: torch.device,
    model_comm_group: dist.ProcessGroup,
) -> None:
    atol, rtol = _tolerances()

    batch_size, num_heads, seq_len, channels = 2, 8, 8, 4
    x_full = torch.arange(
        batch_size * num_heads * seq_len * channels,
        dtype=torch.float32,
        device=device,
    ).reshape(batch_size, num_heads, seq_len, channels)
    seq_split_sizes = get_balanced_partition_sizes(seq_len, world_size)
    seq_shapes = [[seq_shard, num_heads * channels] for seq_shard in seq_split_sizes]
    x_seq_local = torch.split(x_full, seq_split_sizes, dim=2)[rank].contiguous().clone().requires_grad_(True)

    x_heads_local = shard_heads(x_seq_local, shapes=seq_shapes, mgroup=model_comm_group)
    head_split_sizes = get_balanced_partition_sizes(num_heads, world_size)
    x_heads_expected = torch.split(x_full, head_split_sizes, dim=1)[rank].contiguous()
    torch.testing.assert_close(x_heads_local, x_heads_expected, atol=atol, rtol=rtol)

    x_seq_roundtrip = shard_sequence(x_heads_local, shapes=seq_shapes, num_heads=num_heads, mgroup=model_comm_group)
    torch.testing.assert_close(x_seq_roundtrip, x_seq_local.detach(), atol=atol, rtol=rtol)
    x_seq_roundtrip.sum().backward()
    torch.testing.assert_close(
        x_seq_local.grad,
        torch.ones_like(x_seq_local),
        atol=atol,
        rtol=rtol,
    )


def _run_channel_primitives(
    rank: int,
    world_size: int,
    device: torch.device,
    model_comm_group: dist.ProcessGroup,
) -> None:
    atol, rtol = _tolerances()

    if world_size < 2:
        msg = "Channel sharding test requires world_size >= 2."
        raise ValueError(msg)

    # Sequence-parallel local shard with full channels -> channel-parallel local shard with full sequence.
    x_full = torch.arange(2 * 7 * 6, dtype=torch.float32, device=device).reshape(2, 7, 6)
    seq_shapes = get_shard_shapes(x_full, dim=1, model_comm_group=model_comm_group)
    x_seq_local = _split_by_shapes(x_full, dim=1, shard_shapes=seq_shapes, rank=rank).clone().requires_grad_(True)

    x_channel_local = shard_channels(x_seq_local, seq_shapes, model_comm_group)
    x_channel_expected = torch.tensor_split(x_full, world_size, dim=-1)[rank].contiguous()
    torch.testing.assert_close(x_channel_local, x_channel_expected, atol=atol, rtol=rtol)

    x_seq_roundtrip = gather_channels(x_channel_local, seq_shapes, model_comm_group)
    torch.testing.assert_close(x_seq_roundtrip, x_seq_local.detach(), atol=atol, rtol=rtol)

    x_seq_roundtrip.sum().backward()
    torch.testing.assert_close(
        x_seq_local.grad,
        torch.ones_like(x_seq_local),
        atol=atol,
        rtol=rtol,
    )


def _run_rank(
    rank: int,
    world_size: int,
    backend: str,
    suite: str,
    init_file: str,
    deterministic: bool,
    matmul_precision: str,
) -> None:
    _configure_numeric_settings(deterministic=deterministic, matmul_precision=matmul_precision)

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
        if suite == "core":
            _run_core_primitives(rank, world_size, device, group)
        elif suite == "transformer":
            _run_transformer_primitives(rank, world_size, device, group)
        else:
            _run_channel_primitives(rank, world_size, device, group)
        dist.barrier(group=group)
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed graph primitive test worker")
    parser.add_argument("--backend", choices=["gloo", "nccl"], required=True)
    parser.add_argument("--suite", choices=["core", "channels", "transformer"], required=True)
    parser.add_argument("--world-size", type=int, default=2)
    args = parser.parse_args()
    deterministic = _env_flag("ANEMOI_DISTRIBUTED_TEST_DETERMINISTIC", default=True)
    matmul_precision = _float32_matmul_precision(default="highest")

    if deterministic and args.backend == "nccl":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if args.world_size < 2:
        msg = f"world-size must be >= 2, got {args.world_size}"
        raise ValueError(msg)

    if args.backend == "nccl" and (not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size):
        msg = f"NCCL backend requires at least {args.world_size} CUDA devices, " f"found {torch.cuda.device_count()}."
        raise RuntimeError(msg)

    fd, path = tempfile.mkstemp(prefix="anemoi_dist_init_", suffix=".tmp")
    os.close(fd)
    init_file = Path(path)
    try:
        mp.spawn(
            _run_rank,
            args=(
                args.world_size,
                args.backend,
                args.suite,
                str(init_file),
                deterministic,
                matmul_precision,
            ),
            nprocs=args.world_size,
            join=True,
        )
        print(
            f"Distributed {args.suite} suite passed on backend={args.backend}.",
            flush=True,
        )
    finally:
        init_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
