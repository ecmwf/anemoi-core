# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import enum
from abc import ABC
from typing import Literal
from typing import Union

from typing_extensions import Self

from anemoi.training.schedulers.utils import get_closest_key
from anemoi.training.schedulers.utils import get_value_from_closest_key

VALID_STEP_TYPES = Literal["step", "epoch"]
VALID_INCREMENT_TYPE = Union[int, dict[int, int], dict[VALID_STEP_TYPES, dict[int, int]]]


class STEPTYPE(str, enum.Enum):
    step: str = "step"
    epoch: str = "epoch"


class Scheduler(ABC):
    """
    `Scheduler` is an abstract base class for schedulers in anemoi.

    A child scheduler should not store any internal counter, and the behaviour should
    be determinstic based on the `_step` and `_epoch`.

    If incrementing values are needed, use `IncrementMixin` to get the number
    of increments.
    """

    _epoch: int = 0
    _step: int = 0

    # Record of step value for each epoch increment
    _epoch_record: dict[int, int]

    _require_epoch_record: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._epoch_record = {}

    def at(
        self,
        step: int | None = None,
        epoch: int | None = None,
        epoch_record: dict[int, int] | None = None,
    ) -> FrozenStateRecord:  # noqa: F821
        """
        Temporarily hold the scheduler at a specific step and epoch.

        Used as context manager.

        Parameters
        ----------
        step : int, optional
            Step value to override with, by default None
        epoch : int, optional
            Epoch value to override with, by default None
        epoch_record : dict[int, int], optional
            Epoch record value to override with.
            Is a dict of epoch: step, the value of step when the epoch
            was updated. by default None


        Example
        --------
        >>> from anemoi.training.schedulers.schedulers import Scheduler
        >>> scheduler = Scheduler()
        >>> with scheduler.at(step = 10, epoch = 1):
        >>>     print(scheduler._step)
        10
        >>> print(scheduler._step)
        0

        Returns
        -------
        FrozenStateRecord
            Record of the prior state, to be used as a context manager.
        """
        if self._require_epoch_record and epoch_record is None:
            error_msg = (
                "This Scheduler requires an `epoch_record` in order to properly"
                "function when being placed at another state."
            )
            raise RuntimeError(error_msg)

        prior_step = self._step
        prior_epoch = self._epoch
        prior_epoch_record = self._epoch_record

        class FrozenStateRecord:
            """Freeze the state of the Scheduler. Any changes will be reverted on exit."""

            def __enter__(self):
                pass

            def __exit__(context_self, *a):  # noqa: N805
                self._step = prior_step
                self._epoch = prior_epoch
                self._epoch_record = prior_epoch_record

        self._step = step if step is not None else prior_step
        self._epoch = epoch if epoch is not None else prior_epoch
        self._epoch_record = epoch_record if epoch_record is not None else prior_epoch_record

        return FrozenStateRecord()

    def step(self, count: int = 1, /) -> None:
        """Step the scheduler by a count."""
        self._step += count

    def step_epoch(self, count: int = 1, /) -> None:
        """Step the scheduler by a count of epochs."""
        self._epoch += count

        if self._epoch_record is None:
            self._epoch_record = {}

        for e in range(1, count + 1):
            self._epoch_record[self._epoch - (count - e)] = self._step

    def get_state_from(self, obj: Self) -> Self:
        """
        Copy the state of another scheduler into this one.

        Parameters
        ----------
        obj : Self
            Scheduler to get state from

        Returns
        -------
        Self
            Self

        Raises
        ------
        TypeError
            If `obj` not a `Scheduler`
        """
        if not isinstance(obj, Scheduler):
            error_msg = f"Cannot get scheduler state from object of type {type(obj)}."
            raise TypeError(error_msg)

        self._epoch = obj._epoch
        self._step = obj._step
        self._epoch_record = obj._epoch_record

        return self


def resolve_increment_value(
    increment: VALID_INCREMENT_TYPE,
    step: int | None,
    epoch: int | None = None,
    *,
    default_step_type: VALID_STEP_TYPES = STEPTYPE.epoch,
) -> int:
    """
    Resolve value from an increment dictionary based on the step and epoch.

    Parameters
    ----------
    increment : VALID_INCREMENT_TYPE
        Increment dictionary to resolve.
        Can be an int, a dictionary of ints, or a dictionary of dictionaries.
    step : int | None, optional
        Step to resolve at
        by default None
    epoch : int | optional
        Epoch to resolve at
        by default None
    default_step_type : VALID_STEP_TYPES, optional
        Value of either step or epoch to resolve at,
        The associated key must be provided.
        by default STEPTYPE.epoch

    Returns
    -------
    int
        Resolved increment value

    Raises
    ------
    ValueError
        If a step or epoch is not provided when increment dictionary keys are int, or str.
    TypeError
        If increment dictionary cannot be resolved.
    """
    if isinstance(increment, int):
        return increment

    if isinstance(next(iter(increment.keys())), int):
        search_val = step if default_step_type == STEPTYPE.step else epoch
        if search_val is None:
            error_msg = (
                f"As `step_type` was set to {default_step_type}, "
                f"{default_step_type} must be provided when increment dictionary keys are int."
            )
            raise ValueError(error_msg)
        return increment.get(get_closest_key(increment, search_val), 0)

    if isinstance(next(iter(increment.keys())), str):
        increment_step_type = next(iter(increment.keys()))
        if increment_step_type not in STEPTYPE.__members__.values():
            error_msg = "Increment dictionary keys must be either 'step' or 'epoch'."
            raise ValueError(error_msg)

        increment_dict = increment[increment_step_type]
        inc_search_val = step if increment_step_type == STEPTYPE.step else epoch
        if inc_search_val is None:
            error_msg = (
                f"As the increment dictionary uses {increment_step_type}, {increment_step_type} must be provided."
            )
            raise ValueError(error_msg)

        return increment_dict.get(get_closest_key(increment_dict, inc_search_val), 0)

    error_msg = "Increment dictionary keys must be either int or a single str."
    raise TypeError(error_msg)


class IncrementMixin(Scheduler):
    """Mixin class for schedulers that have an incrementing value based on the steps and epochs."""

    _require_epoch_record = True

    def __init__(self, every_n: int, step_type: VALID_STEP_TYPES, increment: VALID_INCREMENT_TYPE = 1, **kwargs):
        super().__init__(**kwargs)

        if step_type not in STEPTYPE.__members__.values():
            error_msg = "Step type must be either 'step' or 'epoch'."
            raise ValueError(error_msg)

        if isinstance(increment, dict) and len(increment) == 0:
            error_msg = (
                "Increment dictionary cannot be empty."
                "\nIt should either be a dictionary of ints or contain a single key of 'step' or 'epoch'."
            )
            raise ValueError(error_msg)

        self._every_n = every_n
        self._step_type = step_type
        self._increment = increment

    def instantaneous_increment(self, step: int, epoch: int) -> int:
        """
        Get increment to apply at a certain step and epoch.

        Does not account for `every_n`.

        Parameters
        ----------
        step : int
            Step to search for
        epoch : int
            Epoch to search for

        Returns
        -------
        int
            Instantaneous increment
        """
        return resolve_increment_value(self._increment, step, epoch, default_step_type=self._step_type)

    def get_total_increment(
        self,
        step: int,
        epoch: int,
        epoch_record: dict[int, int],
        maximum_value: int | None = None,
    ) -> int:
        """
        Get total increment when at a certain step / epoch.

        Using the epoch_record to determine epoch and step relations.

        Parameters
        ----------
        step : int
            Step
        epoch : int
            Epoch
        epoch_record : dict[int, int]
            Record of epoch to step
        maximum_value : int, optional
            Maximum value to allow the increment to reach, by default None

        Returns
        -------
        int
            Total increment
        """
        step_record = {step: epoch for epoch, step in epoch_record.items()}

        total_increment = 0

        value = step if self._step_type == STEPTYPE.step else epoch
        count_of_increments = value // self._every_n

        for count in range(1, count_of_increments + 1):
            current_value = count * self._every_n
            instantaneous_values = {self._step_type: current_value}

            if self._step_type == STEPTYPE.epoch:
                instantaneous_values[STEPTYPE.step] = get_value_from_closest_key(epoch_record, current_value, 0)
            else:
                instantaneous_values[STEPTYPE.epoch] = get_value_from_closest_key(step_record, current_value, 0)

            total_increment = min(maximum_value, total_increment + self.instantaneous_increment(**instantaneous_values))

        return total_increment

    @property
    def total_increment(self) -> int:
        """Current total increment based on step, epoch, and record."""
        return self.get_total_increment(self._step, self._epoch, self._epoch_record)
