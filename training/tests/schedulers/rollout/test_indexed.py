# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest

from anemoi.training.schedulers.rollout.indexed import Lookup
from anemoi.training.schedulers.rollout.indexed import PositionalIndexed


@pytest.mark.parametrize(
    ("rollouts", "num_times_per_element", "test_epoch", "expected"),
    [
        ([1, 2, 3], 1, 0, 1),
        ([1, 2, 3], 1, 1, 2),
        ([1, 2, 3], 2, 0, 1),
        ([1, 2, 3], 2, 2, 2),
        ([4, 5, 6], 1, 0, 4),
        ([4, 5, 6], 1, 1, 5),
        ([4, 5, 6], 2, 0, 4),
        ([4, 5, 6], 2, 2, 5),
    ],
)
def test_positional(rollouts: list[int], num_times_per_element: int, test_epoch: int, expected: int) -> None:
    sched = PositionalIndexed(rollouts, num_times_per_element)
    assert sched.rollout == rollouts[0]
    assert sched.maximum_rollout == max(rollouts)

    assert sched.rollout_at(epoch=test_epoch) == expected


@pytest.mark.parametrize(
    ("rollouts", "test_epoch", "expected"),
    [
        ({0: 1, 1: 2, 2: 3}, 0, 1),
        ({0: 1, 1: 2, 2: 3}, 1, 2),
        ({1: 1, 5: 2}, 1, 1),
        ({1: 1, 5: 2}, 4, 1),
        ({1: 1, 5: 2}, 5, 2),
        ({1: 1, 5: 2}, 10, 2),
        ({5: 2}, 1, 1),
    ],
)
def test_lookup(rollouts: dict[int, int], test_epoch: int, expected: int) -> None:
    sched = Lookup(rollouts)
    assert sched.maximum_rollout == max(rollouts.values())

    assert sched.rollout_at(epoch=test_epoch) == expected
