# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any
from typing import NamedTuple

import numpy as np
import torch


class ModelLayout(NamedTuple):
    """Computed model communication layout for one rank."""

    model_comm_group_ranks: list[np.ndarray]
    model_comm_group_id: int
    model_comm_group_rank: int
    model_comm_num_groups: int


class ReaderLayout(NamedTuple):
    """Computed reader communication layout for one rank."""

    reader_group_ranks: np.ndarray
    reader_group_id: int
    reader_group_rank: int
    reader_group_size: int
    reader_group_root: int


class EnsembleLayout(NamedTuple):
    """Computed ensemble communication layout for one rank."""

    ens_comm_group_ranks: list[np.ndarray]
    ens_comm_subgroup_ranks: list[np.ndarray]
    ens_comm_group_id: int
    ens_comm_group_rank: int
    ens_comm_num_groups: int
    ens_comm_subgroup_id: int
    ens_comm_subgroup_rank: int
    ens_comm_num_subgroups: int
    ens_comm_subgroup_size: int


class EnsembleProcessGroups(NamedTuple):
    """Ensemble process groups created for one rank."""

    ens_comm_groups: list[Any]
    ens_comm_subgroups: list[Any]
    ens_comm_group: Any
    ens_comm_subgroup: Any


def create_model_process_groups(model_comm_group_ranks: list[np.ndarray]) -> list[Any]:
    """Create model communication process groups from rank layouts."""
    return [torch.distributed.new_group(ranks) for ranks in model_comm_group_ranks]


def create_reader_process_groups(reader_group_ranks: np.ndarray) -> list[list[Any]]:
    """Create reader process groups from rank layouts."""
    return [[torch.distributed.new_group(ranks) for ranks in group_ranks] for group_ranks in reader_group_ranks]


def create_ensemble_process_groups(layout: EnsembleLayout) -> EnsembleProcessGroups:
    """Create process groups from ensemble rank layouts."""
    ens_comm_groups = [torch.distributed.new_group(ranks) for ranks in layout.ens_comm_group_ranks]
    ens_comm_subgroups = [torch.distributed.new_group(ranks) for ranks in layout.ens_comm_subgroup_ranks]
    ens_comm_group = ens_comm_groups[layout.ens_comm_group_id]
    ens_comm_subgroup = ens_comm_subgroups[layout.ens_comm_subgroup_id]
    return EnsembleProcessGroups(ens_comm_groups, ens_comm_subgroups, ens_comm_group, ens_comm_subgroup)


def get_my_model_comm_group(num_gpus_per_model: int, global_rank: int, world_size: int) -> tuple[int, int, int]:
    """Determine communication group metadata for a given rank and model group size."""
    model_comm_group_id = global_rank // num_gpus_per_model
    model_comm_group_rank = global_rank % num_gpus_per_model
    model_comm_num_groups = world_size // num_gpus_per_model
    return model_comm_group_id, model_comm_group_rank, model_comm_num_groups


def get_my_ensemble_comm_group(
    num_gpus_per_ensemble: int,
    global_rank: int,
    world_size: int,
) -> tuple[int, int, int]:
    """Determine communication group metadata for a given rank and ensemble group size."""
    return get_my_model_comm_group(num_gpus_per_ensemble, global_rank, world_size)


def get_my_reader_group(
    model_comm_group_rank: int,
    read_group_size: int,
    global_rank: int,
) -> tuple[int, int, int, int]:
    """Determine reader group metadata for a given rank."""
    reader_group_id = model_comm_group_rank // read_group_size
    reader_group_rank = model_comm_group_rank % read_group_size
    reader_group_root = (global_rank // read_group_size) * read_group_size
    return reader_group_id, reader_group_rank, read_group_size, reader_group_root


def build_model_layout(
    world_size: int,
    global_rank: int,
    model_comm_group_size: int,
) -> ModelLayout:
    """Build model rank layouts."""
    assert world_size % model_comm_group_size == 0, (
        f"Total number of GPUs ({world_size}) must be divisible by the number of GPUs "
        f"per model ({model_comm_group_size})."
    )

    model_comm_group_ranks = np.split(
        np.arange(world_size, dtype=int),
        int(world_size / model_comm_group_size),
    )

    model_comm_group_id, model_comm_group_rank, model_comm_num_groups = get_my_model_comm_group(
        model_comm_group_size,
        global_rank,
        world_size,
    )

    return ModelLayout(
        model_comm_group_ranks,
        model_comm_group_id,
        model_comm_group_rank,
        model_comm_num_groups,
    )


def build_reader_layout(
    model_comm_group_ranks: list[np.ndarray],
    model_comm_group_size: int,
    read_group_size: int,
    model_comm_group_rank: int,
    global_rank: int,
) -> ReaderLayout:
    """Build reader rank layouts."""
    assert model_comm_group_size % read_group_size == 0, (
        f"Number of GPUs per model ({model_comm_group_size}) must be divisible by read_group_size "
        f"({read_group_size})."
    )

    reader_group_ranks = np.array(
        [np.split(group_ranks, int(model_comm_group_size / read_group_size)) for group_ranks in model_comm_group_ranks],
    )
    reader_group_id, reader_group_rank, reader_group_size, reader_group_root = get_my_reader_group(
        model_comm_group_rank,
        read_group_size,
        global_rank,
    )

    return ReaderLayout(
        reader_group_ranks,
        reader_group_id,
        reader_group_rank,
        reader_group_size,
        reader_group_root,
    )


def build_ensemble_layout(
    world_size: int,
    global_rank: int,
    ens_comm_group_size: int,
    model_comm_group_size: int,
    model_comm_group_rank: int,
) -> EnsembleLayout:
    """Build ensemble rank layouts."""
    assert world_size % ens_comm_group_size == 0, (
        f"Total number of GPUs ({world_size}) must be divisible by the number of GPUs "
        f"per ensemble ({ens_comm_group_size})."
    )
    assert ens_comm_group_size % model_comm_group_size == 0, (
        f"Number of GPUs per ensemble ({ens_comm_group_size}) must be divisible by the number of GPUs "
        f"per model ({model_comm_group_size})."
    )

    ens_comm_group_ranks = np.split(
        np.arange(world_size, dtype=int),
        int(world_size / ens_comm_group_size),
    )
    ens_comm_group_id, ens_comm_group_rank, ens_comm_num_groups = get_my_ensemble_comm_group(
        ens_comm_group_size,
        global_rank,
        world_size,
    )

    spacing = model_comm_group_size
    ens_comm_subgroup_ranks = [
        group_ranks[offset::spacing] for group_ranks in ens_comm_group_ranks for offset in range(spacing)
    ]

    ens_comm_subgroup_size = ens_comm_group_size // model_comm_group_size
    ens_comm_subgroup_id = ens_comm_group_id * model_comm_group_size + model_comm_group_rank
    ens_comm_subgroup_rank = ens_comm_group_rank // model_comm_group_size
    ens_comm_num_subgroups = world_size // ens_comm_subgroup_size

    return EnsembleLayout(
        ens_comm_group_ranks,
        ens_comm_subgroup_ranks,
        ens_comm_group_id,
        ens_comm_group_rank,
        ens_comm_num_groups,
        ens_comm_subgroup_id,
        ens_comm_subgroup_rank,
        ens_comm_num_subgroups,
        ens_comm_subgroup_size,
    )
