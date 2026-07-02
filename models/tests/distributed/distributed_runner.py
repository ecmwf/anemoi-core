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
import math
import os
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.graph import all_to_all_transpose
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import reduce_shard_tensor
from anemoi.models.distributed.graph import reduce_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.graph import sync_tensor
from communication_primitive_cases import ALL_TO_ALL_TRANSPOSE_CASES
from communication_primitive_cases import CASES_BY_PRIMITIVE
from communication_primitive_cases import PARTITION_CASES

GLOBAL_DEFAULT_ATOL = 1e-12
GLOBAL_DEFAULT_RTOL = 1e-12


def _run_gather_tensor_case(
    rank: int,
    world_size: int,
    device: torch.device,
    model_comm_group: dist.ProcessGroup,
    case_name: str,
    atol: float,
    rtol: float,
) -> None:
    case = PARTITION_CASES[case_name](world_size)
    full = torch.arange(math.prod(case.shape), dtype=torch.float64, device=device).reshape(case.shape)
    sizes = get_balanced_partition_sizes(full.shape[case.dim], world_size)

    # The main-branch API already passes per-rank sizes, so use torch.split directly.
    local = torch.split(full, sizes, dim=case.dim)[rank].contiguous().clone().requires_grad_(True)
    gathered = gather_tensor(local, dim=case.dim, sizes=sizes, mgroup=model_comm_group)

    torch.testing.assert_close(gathered, full, atol=atol, rtol=rtol)

    gathered.sum().backward()
    torch.testing.assert_close(local.grad, torch.ones_like(local), atol=atol, rtol=rtol)


def _run_shard_tensor_case(
    rank: int,
    world_size: int,
    device: torch.device,
    model_comm_group: dist.ProcessGroup,
    case_name: str,
    atol: float,
    rtol: float,
) -> None:
    case = PARTITION_CASES[case_name](world_size)
    full = (
        torch.arange(math.prod(case.shape), dtype=torch.float64, device=device).reshape(case.shape).requires_grad_(True)
    )
    sizes = get_balanced_partition_sizes(full.shape[case.dim], world_size)

    sharded = shard_tensor(full, dim=case.dim, sizes=sizes, mgroup=model_comm_group)
    expected = torch.split(full.detach(), sizes, dim=case.dim)[rank].contiguous()

    torch.testing.assert_close(sharded, expected, atol=atol, rtol=rtol)

    sharded.sum().backward()
    torch.testing.assert_close(full.grad, torch.ones_like(full), atol=atol, rtol=rtol)


def _run_sync_tensor_case(
    rank: int,
    world_size: int,
    device: torch.device,
    model_comm_group: dist.ProcessGroup,
    case_name: str,
    atol: float,
    rtol: float,
) -> None:
    if case_name == "gather_in_fwd_false":
        local = (torch.arange(6, dtype=torch.float64, device=device).reshape(2, 3) + rank).requires_grad_(True)
        sizes = get_balanced_partition_sizes(local.shape[0], world_size)
        synced = sync_tensor(local, dim=0, sizes=sizes, mgroup=model_comm_group, gather_in_fwd=False)

        torch.testing.assert_close(synced, local.detach(), atol=atol, rtol=rtol)

        synced.sum().backward()
        torch.testing.assert_close(local.grad, torch.full_like(local, float(world_size)), atol=atol, rtol=rtol)
        return

    case = PARTITION_CASES[case_name](world_size)
    full = torch.arange(math.prod(case.shape), dtype=torch.float64, device=device).reshape(case.shape)
    sizes = get_balanced_partition_sizes(full.shape[case.dim], world_size)
    local = torch.split(full, sizes, dim=case.dim)[rank].contiguous().clone().requires_grad_(True)

    synced = sync_tensor(local, dim=case.dim, sizes=sizes, mgroup=model_comm_group, gather_in_fwd=True)
    torch.testing.assert_close(synced, full, atol=atol, rtol=rtol)

    synced.sum().backward()
    torch.testing.assert_close(local.grad, torch.full_like(local, float(world_size)), atol=atol, rtol=rtol)


def _run_reduce_shard_tensor_case(
    rank: int,
    world_size: int,
    device: torch.device,
    model_comm_group: dist.ProcessGroup,
    case_name: str,
    atol: float,
    rtol: float,
) -> None:
    case = PARTITION_CASES[case_name](world_size)
    base = torch.arange(math.prod(case.shape), dtype=torch.float64, device=device).reshape(case.shape)
    full = (base + rank).requires_grad_(True)
    sizes = get_balanced_partition_sizes(full.shape[case.dim], world_size)

    reduced_shard = reduce_shard_tensor(full, dim=case.dim, sizes=sizes, mgroup=model_comm_group)
    expected_full = base * world_size + sum(range(world_size))
    expected_shard = torch.split(expected_full, sizes, dim=case.dim)[rank].contiguous()

    torch.testing.assert_close(reduced_shard, expected_shard, atol=atol, rtol=rtol)

    reduced_shard.sum().backward()
    torch.testing.assert_close(full.grad, torch.ones_like(full), atol=atol, rtol=rtol)


def _run_reduce_tensor_case(
    rank: int,
    world_size: int,
    device: torch.device,
    model_comm_group: dist.ProcessGroup,
    case_name: str,
    atol: float,
    rtol: float,
) -> None:
    if case_name != "rank_offset":
        msg = f"Unknown reduce_tensor case: {case_name}"
        raise ValueError(msg)

    local = (torch.arange(6, dtype=torch.float64, device=device).reshape(2, 3) + rank).requires_grad_(True)
    reduced = reduce_tensor(local, mgroup=model_comm_group)
    expected = sum(
        torch.arange(6, dtype=torch.float64, device=device).reshape(2, 3) + other for other in range(world_size)
    )

    torch.testing.assert_close(reduced, expected, atol=atol, rtol=rtol)

    reduced.sum().backward()
    torch.testing.assert_close(local.grad, torch.ones_like(local), atol=atol, rtol=rtol)


def _run_all_to_all_transpose_case(
    rank: int,
    world_size: int,
    device: torch.device,
    model_comm_group: dist.ProcessGroup,
    case_name: str,
    atol: float,
    rtol: float,
) -> None:
    case = ALL_TO_ALL_TRANSPOSE_CASES[case_name](world_size)
    full = torch.arange(math.prod(case.shape), dtype=torch.float64, device=device).reshape(case.shape)
    ndim = len(case.shape)
    split_sizes = get_balanced_partition_sizes(full.shape[case.dim_split % ndim], world_size)
    concat_sizes = get_balanced_partition_sizes(full.shape[case.dim_concat % ndim], world_size)
    local = torch.split(full, concat_sizes, dim=case.dim_concat)[rank].contiguous().clone().requires_grad_(True)

    transposed = all_to_all_transpose(
        local,
        dim_split=case.dim_split,
        split_sizes=split_sizes,
        dim_concat=case.dim_concat,
        concat_sizes=concat_sizes,
        mgroup=model_comm_group,
    )
    expected = torch.split(full, split_sizes, dim=case.dim_split)[rank].contiguous()

    torch.testing.assert_close(transposed, expected, atol=atol, rtol=rtol)

    transposed.sum().backward()
    torch.testing.assert_close(local.grad, torch.ones_like(local), atol=atol, rtol=rtol)


RUNNERS_BY_PRIMITIVE = {
    "all_to_all_transpose": _run_all_to_all_transpose_case,
    "gather_tensor": _run_gather_tensor_case,
    "reduce_shard_tensor": _run_reduce_shard_tensor_case,
    "reduce_tensor": _run_reduce_tensor_case,
    "shard_tensor": _run_shard_tensor_case,
    "sync_tensor": _run_sync_tensor_case,
}


def _run_rank(
    rank: int,
    world_size: int,
    backend: str,
    primitive: str,
    case_name: str,
    init_file: str,
    atol: float,
    rtol: float,
) -> None:
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
        RUNNERS_BY_PRIMITIVE[primitive](rank, world_size, device, group, case_name, atol, rtol)
        dist.barrier(group=group)
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed communication primitive test worker")
    parser.add_argument("--backend", choices=["gloo", "nccl"], required=True)
    parser.add_argument("--primitive", choices=list(CASES_BY_PRIMITIVE), required=True)
    parser.add_argument(
        "--case",
        choices=sorted({case for cases in CASES_BY_PRIMITIVE.values() for case in cases}),
        required=True,
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--atol", type=float, default=GLOBAL_DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=GLOBAL_DEFAULT_RTOL)
    args = parser.parse_args()

    if args.world_size < 2:
        msg = f"world-size must be >= 2, got {args.world_size}"
        raise ValueError(msg)

    if args.case not in CASES_BY_PRIMITIVE[args.primitive]:
        msg = f"Unknown {args.primitive} case: {args.case}"
        raise ValueError(msg)

    if args.backend == "nccl" and (not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size):
        msg = f"NCCL backend requires at least {args.world_size} CUDA devices, found {torch.cuda.device_count()}."
        raise RuntimeError(msg)

    # Use a file:// rendezvous so local spawned ranks can initialize without torchrun, SLURM, or a TCP port.
    fd, path = tempfile.mkstemp(prefix="anemoi_dist_init_", suffix=".tmp")
    os.close(fd)
    init_file = Path(path)
    try:
        mp.spawn(
            _run_rank,
            args=(
                args.world_size,
                args.backend,
                args.primitive,
                args.case,
                str(init_file),
                args.atol,
                args.rtol,
            ),
            nprocs=args.world_size,
            join=True,
        )
        print(
            f"Distributed {args.primitive}[{args.case}] passed on backend={args.backend}.",
            flush=True,
        )
    finally:
        init_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
