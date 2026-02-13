# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.training.train.training_task.base import BaseTask


class BaseExplicitIndicesTask(BaseTask):
    """Base class for tasks with explicit time indices."""

    def __init__(self, input_times: list[int], target_times: list[int], **_kwargs):
        self.input_times = input_times
        self.target_times = target_times

    def get_relative_time_indices(self) -> list[int]:
        """Get the relative time indices for the model input sequence.

        Returns
        -------
            list[int]: List of relative time indices.
        """
        return sorted(set(self.input_times + self.target_times))

    def get_inputs(self, index: int) -> list[int]:
        return [index + t for t in self.input_times]

    def get_targets(self, index: int) -> list[int]:
        return [index + t for t in self.target_times]
