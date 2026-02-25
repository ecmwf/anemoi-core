# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest
import torch

from anemoi.training.distributed.groups import build_ensemble_layout
from anemoi.training.distributed.groups import build_model_layout
from anemoi.training.distributed.groups import build_reader_layout
from anemoi.training.distributed.groups import create_ensemble_process_groups
from anemoi.training.distributed.groups import create_model_process_groups
from anemoi.training.distributed.groups import create_reader_process_groups
from anemoi.training.distributed.groups import get_my_ensemble_comm_group


def _to_rank_tuple(ranks: np.ndarray) -> tuple[int, ...]:
    return tuple(int(rank) for rank in ranks)


def test_build_model_and_reader_layout() -> None:
    model_layout = build_model_layout(
        world_size=8,
        global_rank=5,
        model_comm_group_size=4,
    )
    reader_layout = build_reader_layout(
        model_comm_group_ranks=model_layout.model_comm_group_ranks,
        model_comm_group_size=4,
        read_group_size=2,
        model_comm_group_rank=model_layout.model_comm_group_rank,
        global_rank=5,
    )

    assert model_layout.model_comm_group_id == 1
    assert model_layout.model_comm_group_rank == 1
    assert model_layout.model_comm_num_groups == 2
    assert reader_layout.reader_group_id == 0
    assert reader_layout.reader_group_rank == 1
    assert reader_layout.reader_group_size == 2
    assert reader_layout.reader_group_root == 4

    assert _to_rank_tuple(model_layout.model_comm_group_ranks[0]) == (0, 1, 2, 3)
    assert _to_rank_tuple(model_layout.model_comm_group_ranks[1]) == (4, 5, 6, 7)
    assert np.array_equal(
        reader_layout.reader_group_ranks,
        np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]),
    )


def test_build_model_layout_invalid_model_group_size() -> None:
    with pytest.raises(AssertionError, match="must be divisible by the number of GPUs per model"):
        build_model_layout(
            world_size=10,
            global_rank=0,
            model_comm_group_size=4,
        )


def test_build_reader_layout_invalid_reader_group_size() -> None:
    model_layout = build_model_layout(
        world_size=12,
        global_rank=0,
        model_comm_group_size=6,
    )
    with pytest.raises(AssertionError, match="must be divisible by read_group_size"):
        build_reader_layout(
            model_comm_group_ranks=model_layout.model_comm_group_ranks,
            model_comm_group_size=6,
            read_group_size=4,
            model_comm_group_rank=model_layout.model_comm_group_rank,
            global_rank=0,
        )


def test_create_model_and_reader_process_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    model_layout = build_model_layout(
        world_size=8,
        global_rank=5,
        model_comm_group_size=4,
    )
    reader_layout = build_reader_layout(
        model_comm_group_ranks=model_layout.model_comm_group_ranks,
        model_comm_group_size=4,
        read_group_size=2,
        model_comm_group_rank=model_layout.model_comm_group_rank,
        global_rank=5,
    )

    created_groups: list[tuple[int, ...]] = []

    def _fake_new_group(ranks: np.ndarray) -> str:
        created_groups.append(_to_rank_tuple(ranks))
        return f"group-{len(created_groups) - 1}"

    monkeypatch.setattr(torch.distributed, "new_group", _fake_new_group)

    model_comm_groups = create_model_process_groups(model_layout.model_comm_group_ranks)
    reader_groups = create_reader_process_groups(reader_layout.reader_group_ranks)
    model_reader_groups = reader_groups[model_layout.model_comm_group_id]

    assert created_groups == [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
    ]
    assert model_comm_groups[model_layout.model_comm_group_id] == "group-1"
    assert model_reader_groups == ["group-4", "group-5"]


def test_build_ensemble_layout() -> None:
    layout = build_ensemble_layout(
        world_size=16,
        global_rank=10,
        ens_comm_group_size=8,
        model_comm_group_size=4,
        model_comm_group_rank=2,
    )

    assert layout.ens_comm_group_id == 1
    assert layout.ens_comm_group_rank == 2
    assert layout.ens_comm_num_groups == 2
    assert layout.ens_comm_subgroup_id == 6
    assert layout.ens_comm_subgroup_rank == 0
    assert layout.ens_comm_num_subgroups == 8
    assert layout.ens_comm_subgroup_size == 2

    assert _to_rank_tuple(layout.ens_comm_group_ranks[0]) == (0, 1, 2, 3, 4, 5, 6, 7)
    assert _to_rank_tuple(layout.ens_comm_group_ranks[1]) == (8, 9, 10, 11, 12, 13, 14, 15)
    assert _to_rank_tuple(layout.ens_comm_subgroup_ranks[6]) == (10, 14)


def test_get_my_ensemble_comm_group() -> None:
    ens_comm_group_id, ens_comm_group_rank, ens_comm_num_groups = get_my_ensemble_comm_group(
        num_gpus_per_ensemble=8,
        global_rank=10,
        world_size=16,
    )
    assert ens_comm_group_id == 1
    assert ens_comm_group_rank == 2
    assert ens_comm_num_groups == 2


def test_build_ensemble_layout_invalid_ensemble_size() -> None:
    with pytest.raises(AssertionError, match="must be divisible by the number of GPUs per ensemble"):
        build_ensemble_layout(
            world_size=10,
            global_rank=0,
            ens_comm_group_size=4,
            model_comm_group_size=2,
            model_comm_group_rank=0,
        )


def test_build_ensemble_layout_invalid_model_divisor() -> None:
    with pytest.raises(AssertionError, match="must be divisible by the number of GPUs per model"):
        build_ensemble_layout(
            world_size=12,
            global_rank=0,
            ens_comm_group_size=6,
            model_comm_group_size=4,
            model_comm_group_rank=0,
        )


def test_create_ensemble_process_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    layout = build_ensemble_layout(
        world_size=16,
        global_rank=10,
        ens_comm_group_size=8,
        model_comm_group_size=4,
        model_comm_group_rank=2,
    )

    created_groups: list[tuple[int, ...]] = []

    def _fake_new_group(ranks: np.ndarray) -> str:
        created_groups.append(_to_rank_tuple(ranks))
        return f"group-{len(created_groups) - 1}"

    monkeypatch.setattr(torch.distributed, "new_group", _fake_new_group)

    process_groups = create_ensemble_process_groups(layout)

    assert created_groups == [
        (0, 1, 2, 3, 4, 5, 6, 7),
        (8, 9, 10, 11, 12, 13, 14, 15),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
        (8, 12),
        (9, 13),
        (10, 14),
        (11, 15),
    ]
    assert process_groups.ens_comm_group == "group-1"
    assert process_groups.ens_comm_subgroup == "group-8"
