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
from anemoi.models.distributed.graph import gather_tensor

GLOBAL_DEFAULT_ATOL = 1e-12
GLOBAL_DEFAULT_RTOL = 1e-12


def _gather_tensor_case(case_name: str, world_size: int) -> tuple[tuple[int, ...], int]:
    """Return a full tensor shape and gather dimension for a named gather_tensor case."""
    if case_name == "dim0_even":
        return (2 * world_size, 3), 0
    if case_name == "dim0_uneven":
        return (2 * world_size + 1, 3), 0
    if case_name == "dim1_uneven":
        return (3, 2 * world_size + 1), 1
    if case_name == "negative_dim_uneven":
        return (3, 2 * world_size + 1), -1

    msg = f"Unknown gather_tensor case: {case_name}"
    raise ValueError(msg)


def _run_gather_tensor_case(
    rank: int,
    world_size: int,
    device: torch.device,
    model_comm_group: dist.ProcessGroup,
    case_name: str,
    atol: float,
    rtol: float,
) -> None:
    shape, dim = _gather_tensor_case(case_name, world_size)
    full = torch.arange(math.prod(shape), dtype=torch.float64, device=device).reshape(shape)
    sizes = get_balanced_partition_sizes(full.shape[dim], world_size)

    local = torch.split(full, sizes, dim=dim)[rank].contiguous().clone().requires_grad_(True)
    gathered = gather_tensor(local, dim=dim, sizes=sizes, mgroup=model_comm_group)

    torch.testing.assert_close(gathered, full, atol=atol, rtol=rtol)

    gathered.sum().backward()
    torch.testing.assert_close(local.grad, torch.ones_like(local), atol=atol, rtol=rtol)


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
        if primitive == "gather_tensor":
            _run_gather_tensor_case(rank, world_size, device, group, case_name, atol, rtol)
        else:
            msg = f"Unknown primitive: {primitive}"
            raise ValueError(msg)

        dist.barrier(group=group)
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed communication primitive test worker")
    parser.add_argument("--backend", choices=["gloo", "nccl"], required=True)
    parser.add_argument("--primitive", choices=["gather_tensor"], required=True)
    parser.add_argument(
        "--case",
        choices=["dim0_even", "dim0_uneven", "dim1_uneven", "negative_dim_uneven"],
        required=True,
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--atol", type=float, default=GLOBAL_DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=GLOBAL_DEFAULT_RTOL)
    args = parser.parse_args()

    if args.world_size < 2:
        msg = f"world-size must be >= 2, got {args.world_size}"
        raise ValueError(msg)

    if args.backend == "nccl" and (not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size):
        msg = f"NCCL backend requires at least {args.world_size} CUDA devices, found {torch.cuda.device_count()}."
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
