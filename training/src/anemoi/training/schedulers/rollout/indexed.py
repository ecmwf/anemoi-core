# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from anemoi.training.schedulers.rollout import InterEpochRolloutMixin
from anemoi.training.schedulers.rollout import RolloutScheduler
from anemoi.training.schedulers.schedulers import STEPTYPE
from anemoi.training.schedulers.schedulers import VALID_INCREMENT_TYPE
from anemoi.training.schedulers.schedulers import VALID_STEP_TYPES
from anemoi.training.schedulers.utils import get_closest_key

LOG = logging.getLogger(__name__)


class PositionalIndexed(RolloutScheduler):
    """
    `PositionalIndexed` retrieves the rollout value from a list of rollouts based on the current epoch or step.

    Once the list is exhausted, the rollout will remain at the last value.
    """

    def __init__(
        self,
        rollouts: list[int],
        num_times_per_element: int = 1,
        step_type: VALID_STEP_TYPES = "epoch",
        **kwargs,
    ):
        """
        `PositionalIndexed` retrieves the rollout value from a list of rollouts based on the current epoch or step.

        Once the list is exhausted, the rollout will remain at the last value.

        Parameters
        ----------
        rollouts : list[int]
            List of rollout values.
        num_times_per_element: int, optional
            Number of times to remain at a element, by default 1
        step_type : Literal['step', 'epoch'], optional
            Type of step, either 'epoch' or 'step'.
            by default 'epoch'.

        Example
        -------
        >>> from anemoi.training.schedulers.rollout.indexed import PositionalIndexed
        >>> RollSched = PositionalIndexed(rollouts = [1, 2, 3, 4], num_times_per_element = 2, step_type = 'epoch')
        >>> RollSched.at(epoch = 1).rollout
        1
        >>> RollSched.at(epoch = 2).rollout
        1
        >>> RollSched.at(epoch = 3).rollout
        2
        """
        super().__init__(**kwargs)
        if step_type not in STEPTYPE.__members__.values():
            error_msg = "Invalid step_type. Must be 'epoch' or 'step'."
            raise ValueError(error_msg)

        self._rollouts = rollouts
        self._num_times_per_element = num_times_per_element
        self._step_type = step_type

    @property
    def rollout(self) -> int:
        if self._step_type == STEPTYPE.epoch:
            count = self.count(n_epochs=self._num_times_per_element)
        else:
            count = self.count(n_steps=self._num_times_per_element)
        return self._rollouts[min(len(self._rollouts), count)]

    @property
    def maximum_rollout(self) -> int:
        return max(self._rollouts)

    def description(self) -> str:
        return (
            f"PositionalIndexed with rollouts {self._rollouts} and num_times_per_{self._step_type} "
            f"{self._num_times_per_element}."
        )


class EpochPositionalIndexed(PositionalIndexed):
    """Epoch based PositionalIndexed."""

    def __init__(self, rollouts: list[int]):
        super().__init__(rollouts, step_type=STEPTYPE.epoch)


class StepPositionalIndexed(PositionalIndexed, InterEpochRolloutMixin):
    """Step based PositionalIndexed."""

    def __init__(self, rollouts: list[int], adjust_maximum: VALID_INCREMENT_TYPE = 0):
        LOG.warning(
            "Changing the rollout value within an epoch can cause issues with prefetched "
            "data, and will likely fail with out of index errors."
            "\nIf you wish to enable this ensure that `adjust_maximum` covers the change"
            "in rollout within any epoch.",
        )
        super().__init__(rollouts, step_type=STEPTYPE.step, adjust_maximum=adjust_maximum)


class Lookup(RolloutScheduler):
    """
    `Lookup` retrieves the rollout value from a dictionary of rollouts based on the current epoch or step.

    It will return the closest key that is less than or equal to the current epoch or step.
    """

    def __init__(self, rollouts: dict[int, int], step_type: VALID_STEP_TYPES = "epoch", **kwargs):
        """
        `Lookup` retrieves the rollout value from a dictionary of rollouts based on the current epoch or step.

        It will return the closest key that is less than or equal to the current epoch or step.

        If there is no key lower then the index, defaults to 1.

        Parameters
        ----------
        rollouts : dict[int, int]
            Dictionary of rollouts.
        step_type : Literal['step', 'epoch'], optional
            Type of step, either 'epoch' or 'step'.
            by default 'epoch'

        Example
        -------
        >>> from anemoi.training.schedulers.rollout.indexed import Lookup
        >>> RollSched = Lookup(rollouts = {0: 1, 5: 2, 10: 3}, step_type = 'epoch')
        >>> RollSched.at(epoch = 1).rollout
        1
        >>> RollSched.at(epoch = 5).rollout
        2
        """
        super().__init__(**kwargs)
        self._rollouts = rollouts
        self._step_type = step_type

    @property
    def rollout(self) -> int:
        if self._step_type == "epoch":
            return self._rollouts.get(get_closest_key(self._rollouts, self._epoch), 1)
        if self._step_type == "step":
            return self._rollouts.get(get_closest_key(self._rollouts, self._step), 1)

        error_msg = "Invalid step_type. Must be 'epoch' or 'step'."
        raise ValueError(error_msg)

    @property
    def maximum_rollout(self) -> int:
        return max(self._rollouts.values())

    def description(self) -> str:
        return f"Lookup with rollouts {self._rollouts} based on {self._step_type}."


class EpochLookup(Lookup):
    """Epoch based Lookup."""

    def __init__(self, rollouts: dict[int, int]):
        super().__init__(rollouts, step_type=STEPTYPE.epoch)


class StepLookup(Lookup, InterEpochRolloutMixin):
    """Step based Lookup."""

    def __init__(self, rollouts: dict[int, int], adjust_maximum: VALID_INCREMENT_TYPE = 0):
        LOG.warning(
            "Changing the rollout value within an epoch can cause issues with prefetched "
            "data, and will likely fail with out of index errors."
            "\nIf you wish to enable this ensure that `adjust_maximum` covers the change"
            "in rollout within any epoch.",
        )
        super().__init__(rollouts, step_type=STEPTYPE.step, adjust_maximum=adjust_maximum)
