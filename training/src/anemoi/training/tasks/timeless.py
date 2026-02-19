# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.training.tasks.base import BaseSingleStepTask


class BaseTimelessTask(BaseSingleStepTask):
    """Base class for timeless tasks."""

    def __init__(self, **_kwargs) -> None:
        pass

    def get_relative_time_indices(self, *_args, **_kwargs) -> list[int]:
        """Get the relative time indices for the model input sequence.

        Returns
        -------
            list[int]: List of relative time indices.
        """
        return [0]

    def get_batch_input_time_indices(self, *args, **kwargs) -> list[int]:
        return [0]

    def get_batch_output_time_indices(self, *args, **kwargs) -> list[int]:
        return [0]


class DownscalingTask(BaseTimelessTask):
    """Downscaling task implementation."""

    name: str = "downscaling"


class AutoencodingTask(BaseTimelessTask):
    """Autoencoding task implementation."""

    name: str = "autoencoding"
