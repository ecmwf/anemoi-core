# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import warnings
from typing import Any
from typing import Literal

from anemoi.training.schedulers.rollout import RolloutScheduler


def get_closest_key(dictionary: dict[int, Any], key: int) -> int:
    """
    Get the closest int key in a dictionary to a given key.

    Where the closest key is the one with the smallest absolute difference
    and the key is less than or equal to the given key.

    If no lower key is found, returns -1.

    Parameters
    ----------
    dictionary : dict[int, Any]
        Dictionary to search.
    key : int
        Key to search for.

    Returns
    -------
    int
        Closest key in the dictionary.
    """
    lowest_key = min(dictionary.keys(), key=lambda x: abs(x - key) if x <= key else float("inf"))
    if key < lowest_key:
        return -1
    return lowest_key


class PositionalIndexed(RolloutScheduler):
    """
    `PositionalIndexed` retrieves the rollout value from a list of rollouts based on the current epoch or step.

    Once the list is exhausted, the rollout will remain at the last value.
    """

    def __init__(
        self,
        rollouts: list[int],
        num_times_per_element: int = 1,
        step_type: Literal["step", "epoch"] = "epoch",
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
        ```python
        from anemoi.training.schedulers.rollout.indexed import PositionalIndexed

        RollSched = PositionalIndexed(rollouts = [1, 2, 3, 4], num_times_per_element = 2, step_type = 'epoch')
        RollSched.at_epoch(1)
        # 1
        RollSched.at_epoch(2)
        # 1
        RollSched.at_epoch(3)
        # 2
        ```
        """
        super().__init__()
        self._rollouts = rollouts
        self._num_times_per_element = num_times_per_element
        self._step_type = step_type

    @property
    def rollout(self) -> int:
        if self._step_type == "epoch":
            count = self.count(n_epochs=self._num_times_per_element)
        elif self._step_type == "step":
            count = self.count(n_steps=self._num_times_per_element)
        else:
            error_msg = "Invalid step_type. Must be 'epoch' or 'step'."
            raise ValueError(error_msg)
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
        super().__init__(rollouts, step_type="epoch")


class StepPositionalIndexed(PositionalIndexed):
    """Step based PositionalIndexed."""

    def __init__(self, rollouts: list[int]):
        warnings.warn(
            "Pytorch Lightning datamodules can only be refreshed at the end of an epoch, "
            "adjusting the rollout during an epoch will likely fail.",
            UserWarning,
        )
        super().__init__(rollouts, step_type="step")


class Lookup(RolloutScheduler):
    """
    `Lookup` retrieves the rollout value from a dictionary of rollouts based on the current epoch or step.

    It will return the closest key that is less than or equal to the current epoch or step.
    """

    def __init__(self, rollouts: dict[int, int], step_type: Literal["step", "epoch"] = "epoch"):
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
        ```python
        from anemoi.training.schedulers.rollout.indexed import Lookup

        RollSched = Lookup(rollouts = {0: 1, 5: 2, 10: 3}, step_type = 'epoch')
        RollSched.at_epoch(1)
        # 1
        RollSched.at_epoch(5)
        # 2
        ```
        """
        super().__init__()
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
        super().__init__(rollouts, step_type="epoch")


class StepLookup(Lookup):
    """Step based Lookup."""

    def __init__(self, rollouts: dict[int, int]):
        warnings.warn(
            "Pytorch Lightning datamodules can only be refreshed at the end of an epoch, "
            "adjusting the rollout during an epoch will likely fail.",
            UserWarning,
        )
        super().__init__(rollouts, step_type="step")
