# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from anemoi.training.schedulers.schedulers import IncrementMixin as IncrementMixin
from anemoi.training.schedulers.schedulers import Scheduler

if TYPE_CHECKING:
    from anemoi.training.schedulers.schedulers import VALID_INCREMENT_TYPE


class RolloutScheduler(Scheduler):
    """
    `RolloutScheduler` is an abstract base class for rollout schedulers.

    A rollout scheduler is an object that manages the rollout of a training loop.

    Example
    -------
    >>> RollSched = RolloutScheduler()
    >>> for epoch in range(20):
    >>>    for step in range(100):
    >>>        y = model(x, rollout = RollSched.rollout)
    >>>        RollSched.step()
    >>>     RollSched.step_epoch()

    Override the `rollout` property to implement the rollout calculation,
    and the `maximum_rollout` property to provide the maximum rollout possible.
    """

    @property
    @abstractmethod
    def rollout(self) -> int:
        """Get the current rollout value."""
        error_msg = "`rollout` property not implemented by parent class."
        raise NotImplementedError(error_msg)

    @property
    @abstractmethod
    def maximum_rollout(self) -> int:
        """Get maximum rollout possible."""
        error_msg = "`maximum_rollout` property not implemented by parent class."
        raise NotImplementedError(error_msg)

    @property
    def current_maximum(self) -> int:
        """Get the current maximum rollout value.

        Allows for dataloader to only get the data neccessary.
        Most cases this is just the current rollout.
        """
        return self.rollout

    def __int__(self) -> int:
        """Get rollout value as int."""
        return int(self.rollout)

    def __index__(self) -> int:
        """Get rollout value as index."""
        return int(self.rollout)

    def rollout_at(
        self,
        step: int | None = None,
        epoch: int | None = None,
        epoch_record: dict[int, int] | None = None,
    ) -> int:
        """
        Get the rollout at a specific step and epoch.

        Parameters
        ----------
        step : int, optional
            Step value to override with, by default None
        epoch : int, optional
            Epoch value to override with, by default None
        epoch_record : dict[int, int], optional
            Epoch record to override with, by default None

        Returns
        -------
        int
            Rollout value at the specified step and epoch.
        """
        with self.at(step, epoch, epoch_record=epoch_record):
            return self.rollout

    def count(self, n_steps: int | None = None, n_epochs: int | None = None) -> int:
        """
        Get the count of steps or epochs.

        Parameters
        ----------
        n_steps : int | None, optional
            Number of steps to count, by default None
        n_epochs : int | None, optional
            Number of epochs to count, by default None

        Returns
        -------
        int
            Count of steps or epochs, rounded down.

        Raises
        ------
        ValueError
            If both `n_epochs` and `n_steps` are given, or if neither are given.
        """
        if (n_epochs is not None and n_steps is not None) or (n_epochs is None and n_steps is None):
            error_msg = "Only one of `n_epochs` or `n_steps` can be given."
            raise ValueError(error_msg)

        if n_epochs is not None:
            return self._epoch // n_epochs
        return self._step // n_steps

    @abstractmethod
    def description(self) -> str:
        """Description of the rollout scheduler."""
        error_msg = "`description` method not implemented by parent class."
        raise NotImplementedError(error_msg)

    # Mathematical operations
    def __add__(self, other: int) -> int:
        return self.rollout + other

    def __radd__(self, other: int) -> int:
        return other + self.rollout

    def __sub__(self, other: int) -> int:
        return self.rollout - other

    def __rsub__(self, other: int) -> int:
        return other - self.rollout

    def __rdiv__(self, other: int | float) -> int | float:
        return other / self.rollout

    def __rfloordiv__(self, other: int | float) -> int:
        return other // self.rollout


class Static(RolloutScheduler):
    """`Static` is a rollout scheduler that always returns the same rollout value."""

    def __init__(self, rollout_value: int, **kwargs):
        """
        `Static` is a rollout scheduler that always returns the same rollout value.

        Parameters
        ----------
        rollout_value : int
            Rollout value to return.

        Example
        -------
        >>> from anemoi.training.schedulers.rollout import Static
        >>> RollSched = Static(rollout_value = 5)
        >>> RollSched.rollout_at(epoch = 1)
        5
        >>> RollSched.rollout_at(epoch = 5)
        5
        """
        super().__init__(**kwargs)
        self._rollout_value = rollout_value

    @property
    def rollout(self) -> int:
        return self._rollout_value

    @property
    def maximum_rollout(self) -> int:
        return self._rollout_value

    def description(self) -> str:
        return f"Static rollout value of {self._rollout_value}."


class InterEpochRolloutMixin(RolloutScheduler):
    """Mixin to enable inter epoch rollout changes."""

    def __init__(self, adjust_maximum: VALID_INCREMENT_TYPE, **kwargs):
        """
        Mixin to enable inter epoch rollout changes.

        Adjusts the current maximum by a set value.

        Parameters
        ----------
        adjust_maximum : VALID_INCREMENT_TYPE
            Value to adjust `current_maximum` by
            Can be int, dict[int, int], or dict[Literal["step", "epoch"], dict[int, int]].
            If dictionary, resolved instantaneously in the same way increments are for stepped based rollouts.
            So, maximum's will not accumulate over time.

            If a dict[int, int], the default step_type is 'step'.

        Example
        -------
        >>> from anemoi.training.schedulers.rollout import InterEpochRolloutMixin, Static
        >>> class InterEpochRollout(InterEpochRolloutMixin, Static):
        >>>     pass
        >>> InterEpochRollout(rollout_value = 1, adjust_maximum = 1).current_maximum
        2
        >>> InterEpochRollout(rollout_value = 1, adjust_maximum = {0: 0, 10: 1}).current_maximum
        1
        >>> RollSched = InterEpochRollout(rollout_value = 1, adjust_maximum = {0: 0, 10: 1})
        >>> RollSched.rollout_at(step = 10)
        2
        >>> RollSched.rollout_at(step = 100)
        2
        >>> RollSched = InterEpochRollout(rollout_value = 1, adjust_maximum = {'epoch': {0: 0, 10: 1}})
        >>> RollSched.rollout_at(epoch = 10)
        2
        >>> RollSched.rollout_at(epoch = 100)
        2
        """
        super().__init__(**kwargs)
        self._adjust_maximum = adjust_maximum

    @property
    def current_maximum(self) -> int:
        from anemoi.training.schedulers.schedulers import resolve_increment_value

        return super().current_maximum + resolve_increment_value(
            self._adjust_maximum,
            self._step,
            self._epoch,
            default_step_type="step",
        )
