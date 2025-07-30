# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING

from anemoi.datasets.data import open_dataset
from anemoi.training.data.dataset import AutoencoderNativeGridDataset

from .singledatamodule import AnemoiDatasetsDataModule

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable


class AnemoiAutoencoderDatasetsDataModule(AnemoiDatasetsDataModule):
    """Anemoi Autoencoder Datasets data module for PyTorch Lightning."""

    def relative_date_indices(self) -> list:
        """Determine a list of relative time indices to load for each batch."""
        multi_step = self.config.training.multistep_input
        return [self.timeincrement * mstep for mstep in range(multi_step)]

    def _get_dataset(
        self,
        data_reader: Callable,
        shuffle: bool = True,
        label: str = "generic",
    ) -> AutoencoderNativeGridDataset:

        data_reader = self.add_trajectory_ids(data_reader)  # NOTE: Functionality to be moved to anemoi datasets

        return AutoencoderNativeGridDataset(
            data_reader=data_reader,
            relative_date_indices=self.relative_date_indices(),
            timestep=self.config.data.timestep,
            shuffle=shuffle,
            grid_indices=self.grid_indices,
            label=label,
        )

    @cached_property
    def ds_valid(self) -> AutoencoderNativeGridDataset:
        if not self.config.dataloader.training.end < self.config.dataloader.validation.start:
            LOGGER.warning(
                "Training end date %s is not before validation start date %s.",
                self.config.dataloader.training.end,
                self.config.dataloader.validation.start,
            )
        return self._get_dataset(
            open_dataset(self.config.dataloader.validation),
            shuffle=False,
            label="validation",
        )
