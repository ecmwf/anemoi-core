# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

import warnings
from typing import Literal

from anemoi.training.schedulers.rollout import RolloutScheduler
from anemoi.training.schedulers.rollout.indexed import get_closest_key

VALID_STEP_TYPE = ["step", "epoch"]
VALID_STEP_TYPES = Literal["step", "epoch"]

VALID_INCREMENT_TYPE = int | dict[int, int] | dict[VALID_STEP_TYPES, dict[int, int]]


class IncrementMixin:
    """Mixin class for schedulers that have an incrementing value based on the steps and epochs."""

    def __init__(self, every_n: int, step_type: VALID_STEP_TYPES, increment: VALID_INCREMENT_TYPE = 1):
        super().__init__()

        if step_type not in VALID_STEP_TYPE:
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

    def increment(self, step: int, epoch: int) -> int:
        """
        Get the increment value for a particular step or epoch.

        Relies on the number of steps per epochs to calculate the increment
        when the step_type of the increment is different from the stepper step_type.


        Parameters
        ----------
        step : int
            Step number.
        epoch : int
            Epoch number.

        Returns
        -------
        int
            Increment value.

        Raises
        ------
        ValueError
            If cannot parse the `increment` value given at init.
        """
        count = step // self._every_n if self._step_type == "step" else epoch // self._every_n

        if isinstance(self._increment, int):
            return self._increment * count

        if isinstance(next(iter(self._increment.keys())), int):
            return sum(
                (self._increment.get(get_closest_key(self._increment, i * self._every_n), 0) for i in range(count + 1)),
            )

        if isinstance(next(iter(self._increment.keys())), str):
            increment_step_type = next(iter(self._increment.keys()))
            if increment_step_type not in ["step", "epoch"]:
                error_msg = "Increment dictionary keys must be either 'step' or 'epoch'."
                raise ValueError(error_msg)

            increment_dict = self._increment[increment_step_type]

            if increment_step_type == self._step_type:
                return sum(
                    (
                        increment_dict.get(get_closest_key(increment_dict, i * self._every_n), 0)
                        for i in range(count + 1)
                    ),
                )

            if epoch == 0 or step == 0:
                return 0

            num_steps_per_epoch = step / epoch
            if increment_step_type == "step" and self._step_type == "epoch":
                return sum(
                    increment_dict.get(
                        get_closest_key(increment_dict, (i * self._every_n) * num_steps_per_epoch),
                        0,
                    )
                    for i in range(count)
                )
            if increment_step_type == "epoch" and self._step_type == "step":
                return sum(
                    increment_dict.get(
                        get_closest_key(increment_dict, (i * self._every_n) // num_steps_per_epoch),
                        0,
                    )
                    for i in range(count + 1)
                )

        error_msg = "Increment dictionary keys must be either int or a single str."
        raise TypeError(error_msg)


class Stepped(RolloutScheduler, IncrementMixin):
    """`Stepped` is a base rollout scheduler that steps the rollout value at the end of each n steps or epochs."""

    def __init__(
        self,
        minimum: int,
        maximum: int,
        every_n: int,
        increment: VALID_INCREMENT_TYPE = 1,
        *,
        step_type: VALID_STEP_TYPES = "epoch",
    ):
        """
        `SteppedRollout` is a base rollout scheduler that steps the rollout value at the end of each n steps or epochs.

        Parameters
        ----------
        minimum : int
            Minimum rollout value.
        maximum : int
            Maximum rollout value.
            Can be -1 to indicate no maximum.
        every_n : int
            Number of steps or epochs to step the rollout value.
            If `every_n` is 0, the rollout will stay at `minimum`.
        increment : int | dict[int, int] | dict[Literal['step', 'epoch'], dict[int, int]], optional
            Value to increment the rollout by.
            Can be an int or dictionary, where the keys represent the value of `step_type`
            and the values represent the increment.
            Will round down to the closest key.
            i.e. {0: 1, 10: 2} will increment by 1 until 10, then by 2.
            by default 1.
        step_type : Literal['step', 'epoch'], optional
            Type of step, either 'epoch' or 'step'.
            by default 'epoch'.

        Example
        -------
        ```python
        from anemoi.training.schedulers.rollout.stepped import Stepped

        RollSched = Stepped(minimum = 1, maximum = 10, every_n = 5, increment = 1)
        RollSched.rollout_at(epoch = 2)
        # 1
        RollSched.rollout_at(epoch = 5)
        # 2

        RollSched = Stepped(minimum = 1, maximum = 10, every_n = 5, increment = 2)
        RollSched.rollout_at(epoch = 2)
        # 1
        RollSched.rollout_at(epoch = 5)
        # 3

        RollSched = Stepped(minimum = 1, maximum = 10, every_n = 1, increment = {0: 0, 10: 1})
        RollSched.rollout_at(epoch = 2)
        # 1
        RollSched.rollout_at(epoch = 9)
        # 1
        RollSched.rollout_at(epoch = 10)
        # 2, and then increments of 1

        RollSched = Stepped(
            minimum = 1, maximum = 10, every_n = 1,
            step_type = 'epoch', increment = {'step':{0: 0, 1000: 1}}
        )
        RollSched.rollout_at(epoch = 1, step = 500 )
        # 1
        RollSched.rollout_at(epoch = 2, step = 1000)
        # 2

        ```
        """
        super().__init__(every_n=every_n, step_type=step_type, increment=increment)

        if maximum == -1:
            maximum = float("inf")

        self._minimum = minimum
        self._maximum = maximum

    @property
    def rollout(self) -> int:
        return min(self._maximum, self._minimum + self.increment(self._step, self._epoch))

    @property
    def maximum_rollout(self) -> int:
        if self._every_n == 0:
            return self._minimum
        return self._maximum

    def description(self) -> str:
        return (
            "Stepped rollout scheduler stepping between "
            f"{self._minimum} and {self._maximum} by {self._increment} for every {self._every_n} {self._step_type}/s."
        )


class EpochStepped(Stepped):
    """`EpochStepped` is a rollout scheduler that steps the rollout value at the end of each n epochs."""

    def __init__(self, minimum: int, maximum: int, every_n_epochs: int = 1, increment: VALID_INCREMENT_TYPE = 1):
        """
        `EpochStepped` is a rollout scheduler that steps the rollout value at the end of each n epochs.

        Parameters
        ----------
        minimum : int
            The minimum value for the scheduler.
        maximum : int
            The maximum value for the scheduler.
        every_n_epochs : int, optional
            The number of epochs after which the value is incremented, by default 1.
        increment : int | dict[int, int] | dict[Literal['step', 'epoch'], dict[int, int]], optional
            Value to increment the rollout by.
            Can be an int or dictionary, where the keys represent the value of `step_type`
            and the values represent the increment.
            Will round down to the closest key.
            i.e. {0: 1, 10: 2} will increment by 1 until 10, then by 2.
            by default 1.
        """
        super().__init__(minimum, maximum, every_n_epochs, increment, step_type="epoch")


class StepStepped(Stepped):
    """`StepStepped` is a rollout scheduler that steps the rollout value at the end of each n steps."""

    def __init__(self, minimum: int, maximum: int, every_n_steps: int = 1000, increment: VALID_INCREMENT_TYPE = 1):
        """
        `StepStepped` is a rollout scheduler that steps the rollout value at the end of each n steps.

        Parameters
        ----------
        minimum : int
            The minimum value for the scheduler.
        maximum : int
            The maximum value for the scheduler.
        every_n_steps : int, optional
            The number of steps after which the value is incremented, by default 1000.
        increment : int | dict[int, int] | dict[Literal['step', 'epoch'], dict[int, int]], optional
            Value to increment the rollout by.
            Can be an int or dictionary, where the keys represent the value of `step_type`
            and the values represent the increment.
            Will round down to the closest key.
            i.e. {0: 1, 10: 2} will increment by 1 until 10, then by 2.
            by default 1.
        """
        warnings.warn(
            "Pytorch Lightning datamodules can only be refreshed at the end of an epoch, "
            "adjusting the rollout during an epoch will likely fail.",
            UserWarning,
        )
        super().__init__(minimum, maximum, every_n_steps, increment, step_type="step")
