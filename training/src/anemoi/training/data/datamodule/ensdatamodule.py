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
from typing import TYPE_CHECKING

from anemoi.training.data.dataset import EnsNativeGridDataset

from .singledatamodule import AnemoiDatasetsDataModule

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable


class AnemoiEnsDatasetsDataModule(AnemoiDatasetsDataModule):
    """Anemoi Ensemble data module for PyTorch Lightning."""

    def _get_dataset(
        self,
        data_reader: Callable,
        shuffle: bool = True,
        val_rollout: int = 1,
        label: str = "generic",
    ) -> EnsNativeGridDataset:

        data_reader = self.add_trajectory_ids(data_reader)  # NOTE: Functionality to be moved to anemoi datasets

        return EnsNativeGridDataset(
            data_reader=data_reader,
            relative_date_indices=self.relative_date_indices(val_rollout),
            shuffle=shuffle,
            grid_indices=self.grid_indices,
            label=label,
            ens_members_per_device=self.config.training.ensemble_size_per_device,
            num_gpus_per_ens=self.config.hardware.num_gpus_per_ensemble,
            num_gpus_per_model=self.config.hardware.num_gpus_per_model,
        )
