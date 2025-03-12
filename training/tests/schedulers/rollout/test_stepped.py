# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from anemoi.training.schedulers.rollout.stepped import EpochStepped
from anemoi.training.schedulers.rollout.stepped import Stepped

STEPPED_TEST = [
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
]


@pytest.mark.parametrize(("minimum", "maximum", "every_n", "increment", "epoch_test", "expected_value"), STEPPED_TEST)
def test_stepped(
    minimum: int,
    maximum: int,
    every_n: int,
    increment: int,
    epoch_test: int,
    expected_value: int,
) -> None:
    sched = Stepped(minimum, maximum, every_n, increment=increment)

    with sched.at(epoch=epoch_test, epoch_record={}):
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
    ("every_n", "increment", "epoch_test", "expected_value"),
    [*COMPLEX_INCREMENT_TESTS_EVERY_N_1, *COMPLEX_INCREMENT_TESTS_EVERY_N_2],
)
def test_stepped_complex_increment(
    every_n: int,
    increment: dict[int, int],
    epoch_test: int,
    expected_value: int,
) -> None:

    sched = Stepped(1, 10, every_n, increment=increment)

    with sched.at(epoch=epoch_test, epoch_record={}):
        assert sched.rollout == expected_value
        assert sched.current_maximum == expected_value


def test_step_per_epoch_change() -> None:
    states = [
        # Epoch, Step, Expected
        (1, 100, 1),
        (2, 200, 1),
        (3, 300, 1),
        (4, 400, 1),
        (5, 500, 2),
        (6, 510, 3),
        (7, 520, 4),
        (8, 530, 5),
        (9, 1000, 6),
    ]
    roll_sched = EpochStepped(1, 1000, every_n_epochs=1, increment={"step": {0: 0, 500: 1}})

    step_record = 0
    epoch_record = 0

    for e, s, ex in states:
        roll_sched.step(s - step_record)
        roll_sched.step_epoch(e - epoch_record)

        step_record = s
        epoch_record = e

        assert ex == roll_sched.rollout
        assert e in roll_sched._epoch_record


def test_step_negative_stepping() -> None:
    states = [
        # Epoch, Step, Expected
        (1, 100, 1),
        (2, 200, 1),
        (3, 300, 11),
        (4, 400, 20),
        (5, 500, 20),
        (6, 600, 20),
        (7, 700, 18),
        (8, 800, 16),
        (9, 900, 14),
        (10, 1000, 12),
        (100, 100 * 100, 1),
    ]

    roll_sched = Stepped(
        minimum=1,
        maximum=20,
        every_n=1,
        increment={"step": {0: 0, 300: 10, 700: -2}},
        step_type="epoch",
    )

    step_record = 0
    epoch_record = 0

    for e, s, ex in states:
        roll_sched.step(s - step_record)
        roll_sched.step_epoch(e - epoch_record)

        step_record = s
        epoch_record = e

        assert ex == roll_sched.rollout
        assert e in roll_sched._epoch_record
