# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from anemoi.training.schedulers.rollout.stepped import Stepped


@pytest.mark.parametrize(
    ("minimum", "maximum", "every_n", "increment", "epoch_test", "expected_value"),
    [
        # Increment of 1 and every_n of 1
        (1, 10, 1, 1, 0, 1),
        (1, 10, 1, 1, 1, 2),
        (1, 10, 1, 1, 5, 6),
        (1, 10, 1, 1, 6, 7),
        (1, 10, 1, 1, 8, 9),
        (1, 10, 1, 1, 9, 10),
        (1, 10, 1, 1, 10, 10),
        (1, 10, 1, 1, 11, 10),
        (1, 10, 1, 1, 1000, 10),
        # Increment of 2 and every_n of 1
        (1, 10, 1, 2, 1, 3),
        (1, 10, 1, 2, 2, 5),
        (1, 10, 1, 2, 4, 9),
        (1, 10, 1, 2, 5, 10),
        # Increment of 1 and every_n of 2
        (1, 10, 2, 1, 0, 1),
        (1, 10, 2, 1, 1, 1),
        (1, 10, 2, 1, 2, 2),
    ],
)
def test_stepped(
    minimum: int,
    maximum: int,
    every_n: int,
    increment: int,
    epoch_test: int,
    expected_value: int,
) -> None:
    sched = Stepped(minimum, maximum, every_n, increment=increment)

    sched.sync(epoch=epoch_test)
    assert sched.rollout == expected_value
    assert sched.current_maximum == expected_value


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
def test_stepped_complex_increment(
    every_n: int,
    increment: dict[int, int],
    epoch_test: int,
    expected_value: int,
) -> None:

    sched = Stepped(1, 10, every_n, increment=increment)

    sched.sync(epoch=epoch_test)
    assert sched.rollout == expected_value
    assert sched.current_maximum == expected_value
