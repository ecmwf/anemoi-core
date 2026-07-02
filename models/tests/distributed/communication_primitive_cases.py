# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PartitionCase:
    shape: tuple[int, ...]
    dim: int


@dataclass(frozen=True)
class AllToAllTransposeCase:
    shape: tuple[int, ...]
    dim_split: int
    dim_concat: int


def dim0_even(world_size: int) -> PartitionCase:
    return PartitionCase(shape=(2 * world_size, 3), dim=0)


def dim0_uneven(world_size: int) -> PartitionCase:
    return PartitionCase(shape=(2 * world_size + 1, 3), dim=0)


def dim1_uneven(world_size: int) -> PartitionCase:
    return PartitionCase(shape=(3, 2 * world_size + 1), dim=1)


def negative_dim_uneven(world_size: int) -> PartitionCase:
    return PartitionCase(shape=(3, 2 * world_size + 1), dim=-1)


def dim0_to_dim1_even(world_size: int) -> AllToAllTransposeCase:
    return AllToAllTransposeCase(shape=(2 * world_size, 3 * world_size), dim_split=0, dim_concat=1)


def dim0_to_dim1_uneven(world_size: int) -> AllToAllTransposeCase:
    return AllToAllTransposeCase(shape=(2 * world_size + 1, 3 * world_size + 1), dim_split=0, dim_concat=1)


def negative_dims_uneven(world_size: int) -> AllToAllTransposeCase:
    return AllToAllTransposeCase(shape=(2 * world_size + 1, 3 * world_size + 1), dim_split=-2, dim_concat=-1)


PartitionCaseFactory = Callable[[int], PartitionCase]
AllToAllTransposeCaseFactory = Callable[[int], AllToAllTransposeCase]

PARTITION_CASES: dict[str, PartitionCaseFactory] = {
    "dim0_even": dim0_even,
    "dim0_uneven": dim0_uneven,
    "dim1_uneven": dim1_uneven,
    "negative_dim_uneven": negative_dim_uneven,
}
ALL_TO_ALL_TRANSPOSE_CASES: dict[str, AllToAllTransposeCaseFactory] = {
    "dim0_to_dim1_even": dim0_to_dim1_even,
    "dim0_to_dim1_uneven": dim0_to_dim1_uneven,
    "negative_dims_uneven": negative_dims_uneven,
}
REDUCE_TENSOR_CASES = ("rank_offset",)
SYNC_TENSOR_NO_GATHER_CASES = ("gather_in_fwd_false",)

PARTITION_CASE_IDS = tuple(PARTITION_CASES)
ALL_TO_ALL_TRANSPOSE_CASE_IDS = tuple(ALL_TO_ALL_TRANSPOSE_CASES)

CASES_BY_PRIMITIVE = {
    "all_to_all_transpose": ALL_TO_ALL_TRANSPOSE_CASE_IDS,
    "gather_tensor": PARTITION_CASE_IDS,
    "reduce_shard_tensor": PARTITION_CASE_IDS,
    "reduce_tensor": REDUCE_TENSOR_CASES,
    "shard_tensor": PARTITION_CASE_IDS,
    "sync_tensor": PARTITION_CASE_IDS + SYNC_TENSOR_NO_GATHER_CASES,
}
