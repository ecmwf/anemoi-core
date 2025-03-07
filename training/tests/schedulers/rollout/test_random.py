# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any
from unittest.mock import patch

import pytest

from anemoi.training.schedulers.rollout.randomise import IncreasingRandom
from anemoi.training.schedulers.rollout.randomise import RandomList
from anemoi.training.schedulers.rollout.randomise import RandomRange


def test_determism() -> None:
    sched = RandomList([1, 2, 3])
    sched_1 = RandomList([1, 2, 3])

    sched.rollout  # Force a retrieval to try and break the determinism

    for i in range(20):
        with sched.at(epoch=i) and sched_1.at(epoch=i):
            assert sched.rollout == sched_1.rollout


@pytest.mark.parametrize(
    "rollouts",
    [
        [1, 2, 3],
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [16, 2, 3, 4, 5],
    ],
)
@patch("anemoi.training.schedulers.rollout.randomise.BaseRandom._randomly_pick", wraps=RandomList([0])._randomly_pick)
def test_random_list(pick_mock: Any, rollouts: list[int]) -> None:
    sched = RandomList(rollouts)
    assert sched.rollout in rollouts
    assert sched.maximum_rollout == max(rollouts)

    pick_mock.assert_called_once_with(rollouts)


@pytest.mark.parametrize(
    ("minimum", "maximum", "step"),
    [
        (1, 10, 1),
        (1, 10, 2),
        (1, 10, 3),
    ],
)
@patch("anemoi.training.schedulers.rollout.randomise.BaseRandom._randomly_pick", wraps=RandomList([0])._randomly_pick)
def test_random_range(pick_mock: Any, minimum: int, maximum: int, step: int) -> None:
    sched = RandomRange(minimum, maximum, step)
    assert sched.rollout in range(minimum, maximum + 1, step)
    assert sched.maximum_rollout == max(range(minimum, maximum + 1, step))

    pick_mock.assert_called_once_with(range(minimum, maximum + 1, step))


@pytest.mark.parametrize(
    ("minimum", "maximum", "step", "every_n", "epoch_test", "expected_max"),
    [
        (1, 10, 1, 1, 0, 1),
        (1, 10, 1, 1, 1, 2),
        (1, 10, 1, 1, 2, 3),
        (1, 10, 1, 1, 10, 10),
        (1, 10, 1, 1, 100, 10),
        (1, 10, 1, 2, 2, 2),
        (1, 10, 1, 2, 4, 3),
        (1, 10, 2, 2, 4, 3),
    ],
)
@patch("anemoi.training.schedulers.rollout.randomise.BaseRandom._randomly_pick", wraps=RandomList([0])._randomly_pick)
def test_increasing_random_increment(
    pick_mock: Any,
    minimum: int,
    maximum: int,
    step: int,
    every_n: int,
    epoch_test: int,
    expected_max: int,
) -> None:
    sched = IncreasingRandom(minimum, maximum, step, every_n, 1)

    with sched.at(epoch=epoch_test, epoch_record={}):

        assert sched.current_maximum == expected_max
        assert sched.rollout in list(range(minimum, expected_max + 1, step))
        assert sched.maximum_rollout == maximum

        pick_mock.assert_called_once_with(range(minimum, expected_max + 1, step))


INCREMENT_DICT = {
    0: 0,
    2: 1,
    4: 2,
}
INCREMENT_DICT_1 = {
    0: 0,
    2: 1,
    3: 0,
    4: 2,
}

COMPLEX_INCREMENT_TESTS_EVERY_N_1 = [
    (1, INCREMENT_DICT, 0, 1),
    (1, INCREMENT_DICT, 1, 1),
    (1, INCREMENT_DICT, 2, 2),
    (1, INCREMENT_DICT, 3, 3),
    (1, INCREMENT_DICT, 4, 5),
    (1, INCREMENT_DICT, 5, 7),
    (1, INCREMENT_DICT_1, 4, 4),
    (1, INCREMENT_DICT_1, 5, 6),
    (1, INCREMENT_DICT_1, 1000, 10),
]

COMPLEX_INCREMENT_TESTS_EVERY_N_2 = [
    (2, INCREMENT_DICT, 0, 1),
    (2, INCREMENT_DICT, 1, 1),
    (2, INCREMENT_DICT, 2, 2),
    (2, INCREMENT_DICT, 3, 2),
    (2, INCREMENT_DICT, 4, 4),
    (2, INCREMENT_DICT, 5, 4),
    (2, INCREMENT_DICT, 6, 6),
]


@pytest.mark.parametrize(
    ("every_n", "increment", "epoch_test", "expected_max"),
    [*COMPLEX_INCREMENT_TESTS_EVERY_N_1, *COMPLEX_INCREMENT_TESTS_EVERY_N_2],
)
@patch("anemoi.training.schedulers.rollout.randomise.BaseRandom._randomly_pick", wraps=RandomList([0])._randomly_pick)
def test_increasing_random_complex_increment(
    pick_mock: Any,
    every_n: int,
    increment: dict[int, int],
    epoch_test: int,
    expected_max: int,
) -> None:
    sched = IncreasingRandom(1, 10, 1, every_n, increment=increment)

    with sched.at(epoch=epoch_test, epoch_record={}):
        assert sched.rollout in list(range(1, expected_max + 1, 1))
        assert sched.current_maximum == expected_max
        pick_mock.assert_called_with(range(1, expected_max + 1, 1))
