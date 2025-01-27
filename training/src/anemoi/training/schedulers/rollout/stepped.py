# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

import warnings

from anemoi.training.schedulers.rollout import InterEpochRolloutMixin
from anemoi.training.schedulers.rollout import RolloutScheduler
from anemoi.training.schedulers.schedulers import STEPTYPE
from anemoi.training.schedulers.schedulers import VALID_INCREMENT_TYPE
from anemoi.training.schedulers.schedulers import VALID_STEP_TYPES
from anemoi.training.schedulers.schedulers import IncrementMixin


class Stepped(RolloutScheduler, IncrementMixin):
    """`Stepped` is a base rollout scheduler that steps the rollout value at the end of each n steps or epochs."""

    def __init__(
        self,
        minimum: int,
        maximum: int,
        every_n: int,
        increment: VALID_INCREMENT_TYPE = 1,
        *,
        step_type: VALID_STEP_TYPES = STEPTYPE.epoch,
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

        RollSched = Stepped(minimum = 1, maximum = 10, every_n = 5, increment = 1, step_type = "epoch")
        RollSched.rollout
        # 1
        RollSched.at(epoch = 5, epoch_record = {})
        # 2

        RollSched = Stepped(minimum = 1, maximum = 10, every_n = 1, increment = {0: 0, 10: 1}, step_type = "epoch")
        RollSched.at(epoch = 2, epoch_record = {}).rollout
        # 1
        RollSched.at(epoch = 10, epoch_record = {}).rollout
        # 2, and then increments of 1
        ```
        """
        super().__init__(every_n=every_n, step_type=step_type, increment=increment)

        if maximum == -1:
            maximum = float("inf")

        self._minimum = minimum
        self._maximum = maximum

    @property
    def rollout(self) -> int:
        return min(self._maximum, self._minimum + self.total_increment)

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
        super().__init__(minimum, maximum, every_n_epochs, increment, step_type=STEPTYPE.epoch)


class StepStepped(Stepped, InterEpochRolloutMixin):
    """`StepStepped` is a rollout scheduler that steps the rollout value at the end of each n steps."""

    def __init__(
        self,
        minimum: int,
        maximum: int,
        every_n_steps: int = 1000,
        increment: VALID_INCREMENT_TYPE = 1,
        *,
        adjust_maximum: int = 0,
    ):
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
        adjust_maximum : int, optional
            Value to adjust current maximum by, by default 0
        """
        warnings.warn(
            "Pytorch Lightning datamodules can only be refreshed at the end of an epoch, "
            "adjusting the rollout during an epoch will likely fail."
            "\nIf you wish to enable this ensure that `adjust_maximum` covers the change"
            "in rollout within any epoch.",
            UserWarning,
        )
        super().__init__(
            minimum,
            maximum,
            every_n_steps,
            increment,
            step_type=STEPTYPE.step,
            adjust_maximum=adjust_maximum,
        )
