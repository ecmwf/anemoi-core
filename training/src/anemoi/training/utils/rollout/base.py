# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

from abc import abstractmethod


class BaseRollout:
    """
    `BaseRollout` is an abstract base class for rollout schedulers.

    A rollout scheduler is an object that manages the rollout of a training loop.

    Example
    -------
    >>> RollSched = BaseRollout()
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
    def maximum_rollout(self) -> int:
        """Get maximum rollout possible."""

    @abstractmethod
    def rollout(self, epoch: int) -> int:
        """Get the current rollout value."""
        return self.rollout(epoch)

    @abstractmethod
    def description(self) -> str:
        """Description of the rollout scheduler."""


class Static(BaseRollout):
    """`Static` is a rollout scheduler that always returns the same rollout value."""

    def __init__(self, value: int, **kwargs):
        """
        `Static` is a rollout scheduler that always returns the same rollout value.

        Parameters
        ----------
        value : int
            Rollout value to return.

        Example
        -------
        >>> from anemoi.training.utils.rollout import Static
        >>> RollSched = Static(value = 5)
        >>> RollSched.rollout_at(epoch = 1)
        5
        >>> RollSched.rollout_at(epoch = 5)
        5
        """
        super().__init__(**kwargs)
        self.value = value

    def rollout(self, epoch: int) -> int:
        return self.value

    def description(self) -> str:
        return f"Static rollout value of {self.value}."
