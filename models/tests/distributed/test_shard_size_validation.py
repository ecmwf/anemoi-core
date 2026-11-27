# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the shard-size / process-group contract.

These run in a single process: the guards under test only inspect ``group.size()`` and are
supposed to reject bad input *before* any collective is issued, so no spawned ranks (and no
``--distributed``) are needed. The actual data movement is covered by
``test_communication_primitives.py`` and ``test_data_view_allgather.py``.
"""

from __future__ import annotations

import pytest
import torch

from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import check_shard_sizes_match_group


class _FakeGroup:
    """Minimal ProcessGroup stand-in; the guards only call ``size()``."""

    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


@pytest.mark.parametrize("group", [None, _FakeGroup(1)], ids=["no_group", "single_rank_group"])
def test_gather_tensor_is_noop_for_absent_and_single_rank_groups(group: _FakeGroup | None) -> None:
    """``sizes=None`` must stay legal on a non-distributed group.

    ``DDPGroupStrategy`` creates a real 1-rank process group when ``num_gpus_per_model=1``,
    and several mapper/decoder callsites pass ``sizes=None`` there, relying on the bypass.
    """
    x = torch.arange(6.0).reshape(3, 2)

    assert gather_tensor(x, dim=0, sizes=None, mgroup=group) is x
    assert shard_tensor(x, dim=0, sizes=None, mgroup=group) is x


def test_gather_tensor_rejects_none_sizes_when_group_is_distributed() -> None:
    x = torch.arange(6.0).reshape(3, 2)

    with pytest.raises(ValueError, match="shard_sizes=None"):
        gather_tensor(x, dim=0, sizes=None, mgroup=_FakeGroup(2))


def test_shard_tensor_rejects_none_sizes_when_group_is_distributed() -> None:
    x = torch.arange(6.0).reshape(3, 2)

    with pytest.raises(ValueError, match="shard_sizes=None"):
        shard_tensor(x, dim=0, sizes=None, mgroup=_FakeGroup(2))


def test_gather_tensor_rejects_shard_sizes_from_another_group() -> None:
    """A reader-group-sized descriptor gathered over the model comm group must fail loudly."""
    x = torch.arange(6.0).reshape(3, 2)

    with pytest.raises(ValueError, match="Shard/process-group mismatch"):
        gather_tensor(x, dim=0, sizes=[2, 2], mgroup=_FakeGroup(4))


@pytest.mark.parametrize(
    ("shard_sizes", "group"),
    [
        (None, None),
        (None, _FakeGroup(4)),
        ([3], None),
        ([3], _FakeGroup(1)),
        ([2, 1], _FakeGroup(2)),
    ],
)
def test_check_shard_sizes_match_group_accepts_none_and_matching_sizes(
    shard_sizes: list[int] | None,
    group: _FakeGroup | None,
) -> None:
    check_shard_sizes_match_group(shard_sizes, group)  # must not raise


def test_check_shard_sizes_match_group_names_reader_group_confusion() -> None:
    with pytest.raises(ValueError, match="read_group_size") as excinfo:
        check_shard_sizes_match_group([2, 2], _FakeGroup(4), context="era5")

    message = str(excinfo.value)
    assert "era5" in message
    assert "2 entries" in message
    assert "4 rank(s)" in message
