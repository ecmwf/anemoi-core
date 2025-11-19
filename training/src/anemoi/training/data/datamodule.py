# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import Callable
from functools import cached_property

import numpy as np
import pytorch_lightning as pl
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData

from anemoi.datasets import open_dataset
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.data.dataset import NativeGridDataset
from anemoi.training.data.grid_indices import BaseGridIndices
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.utils.worker_init import worker_init_func
from anemoi.utils.dates import frequency_to_seconds

LOGGER = logging.getLogger(__name__)


class AnemoiDatasetsDataModule(pl.LightningDataModule):
    """Anemoi Datasets data module for PyTorch Lightning."""

    def __init__(self, config: BaseSchema, graph_data: HeteroData) -> None:
        """Initialize Anemoi Datasets data module.

        Parameters
        ----------
        config : BaseSchema
            Job configuration

        """
        super().__init__()

        self.config = config
        self.graph_data = graph_data

        # Set the training end date if not specified
        if self.config.dataloader.training.end is None:
            LOGGER.info(
                "No end date specified for training data, setting default before validation start date %s.",
                self.config.dataloader.validation.start - 1,
            )
            self.config.dataloader.training.end = self.config.dataloader.validation.start - 1

        if not self.config.dataloader.pin_memory:
            LOGGER.info("Data loader memory pinning disabled.")

    @cached_property
    def statistics(self) -> dict:
        return self.ds_train.statistics

    @cached_property
    def statistics_tendencies(self) -> dict:
        return self.ds_train.statistics_tendencies

    @cached_property
    def metadata(self) -> dict:
        return self.ds_train.metadata

    @cached_property
    def supporting_arrays(self) -> dict:
        return self.ds_train.supporting_arrays | self.grid_indices.supporting_arrays

    @cached_property
    def data_indices(self) -> IndexCollection:
        return IndexCollection(self.config, self.ds_train.name_to_index)

    def relative_date_indices(self, val_rollout: int = 1) -> list:
        """Determine a list of relative time indices to load for each batch."""
        if hasattr(self.config.training, "explicit_times"):
            return sorted(set(self.config.training.explicit_times.input + self.config.training.explicit_times.target))

        # Calculate indices using multistep, timeincrement and rollout.
        # Use the maximum rollout to be expected
        rollout_cfg = getattr(getattr(self.config, "training", None), "rollout", None)

        rollout_max = getattr(rollout_cfg, "max", None)
        rollout_start = getattr(rollout_cfg, "start", 1)
        rollout_epoch_increment = getattr(rollout_cfg, "epoch_increment", 0)

        # Fallback if max is None or rollout_cfg is missing
        rollout_value = rollout_start
        if rollout_cfg and rollout_epoch_increment > 0 and rollout_max is not None:
            rollout_value = rollout_max

        else:
            LOGGER.warning(
                "Falling back rollout to: %s",
                rollout_value,
            )

        rollout = max(rollout_value, val_rollout)

        multi_step = self.config.training.multistep_input
        return [self.timeincrement * mstep for mstep in range(multi_step + rollout)]

    def add_trajectory_ids(self, data_reader: Callable) -> Callable:
        """Determine an index of forecast trajectories associated with the time index and add to a data_reader object.

        This is needed for interpolation to ensure that the interpolator is trained on consistent time slices.

        NOTE: This is only relevant when training on non-analysis and could in the future be replaced with
        a property of the dataset stored in data_reader. Now assumes regular interval of changed model runs
        """
        if not hasattr(self.config.dataloader, "model_run_info"):
            data_reader.trajectory_ids = None
            return data_reader

        mr_start = np.datetime64(self.config.dataloader.model_run_info.start)
        mr_len = self.config.dataloader.model_run_info.length  # model run length in number of date indices
        if hasattr(self.config.training, "rollout") and self.config.training.rollout.max is not None:
            max_rollout_index = max(self.relative_date_indices(self.config.training.rollout.max))
            assert (
                max_rollout_index < mr_len
            ), f"""Requested data length {max_rollout_index + 1}
                    longer than model run length {mr_len}"""

        data_reader.trajectory_ids = (data_reader.dates - mr_start) // np.timedelta64(
            mr_len * frequency_to_seconds(self.config.data.frequency),
            "s",
        )
        return data_reader

    @cached_property
    def grid_indices(self) -> type[BaseGridIndices]:
        reader_group_size = self.config.dataloader.read_group_size

        grid_indices = instantiate(
            self.config.dataloader.grid_indices,
            reader_group_size=reader_group_size,
        )
        grid_indices.setup(self.graph_data)
        return grid_indices

    @cached_property
    def timeincrement(self) -> int:
        """Determine the step size relative to the data frequency."""
        try:
            frequency = frequency_to_seconds(self.config.data.frequency)
        except ValueError as e:
            msg = f"Error in data frequency, {self.config.data.frequency}"
            raise ValueError(msg) from e

        try:
            timestep = frequency_to_seconds(self.config.data.timestep)
        except ValueError as e:
            msg = f"Error in timestep, {self.config.data.timestep}"
            raise ValueError(msg) from e

        assert timestep % frequency == 0, (
            f"Timestep ({self.config.data.timestep} == {timestep}) isn't a "
            f"multiple of data frequency ({self.config.data.frequency} == {frequency})."
        )

        LOGGER.info(
            "Timeincrement set to %s for data with frequency, %s, and timestep, %s",
            timestep // frequency,
            frequency,
            timestep,
        )
        return timestep // frequency

    @cached_property
    def ds_train(self) -> NativeGridDataset:
        return self._get_dataset(
            open_dataset(self.config.dataloader.training),
            label="train",
        )

    @cached_property
    def ds_valid(self) -> NativeGridDataset:
        if not self.config.dataloader.training.end < self.config.dataloader.validation.start:
            LOGGER.warning(
                "Training end date %s is not before validation start date %s.",
                self.config.dataloader.training.end,
                self.config.dataloader.validation.start,
            )
        return self._get_dataset(
            open_dataset(self.config.dataloader.validation),
            shuffle=False,
            val_rollout=self.config.dataloader.validation_rollout,
            label="validation",
        )

    @cached_property
    def ds_test(self) -> NativeGridDataset:
        assert self.config.dataloader.training.end < self.config.dataloader.test.start, (
            f"Training end date {self.config.dataloader.training.end} is not before"
            f"test start date {self.config.dataloader.test.start}"
        )
        assert self.config.dataloader.validation.end < self.config.dataloader.test.start, (
            f"Validation end date {self.config.dataloader.validation.end} is not before"
            f"test start date {self.config.dataloader.test.start}"
        )
        return self._get_dataset(
            open_dataset(self.config.dataloader.test),
            shuffle=False,
            label="test",
        )

    def _get_dataset(
        self,
        data_reader: Callable,
        shuffle: bool = True,
        val_rollout: int = 1,
        label: str = "generic",
    ) -> NativeGridDataset:

        data_reader = self.add_trajectory_ids(data_reader)  # NOTE: Functionality to be moved to anemoi datasets

        return NativeGridDataset(
            data_reader=data_reader,
            relative_date_indices=self.relative_date_indices(val_rollout),
            timestep=self.config.data.timestep,
            shuffle=shuffle,
            grid_indices=self.grid_indices,
            label=label,
            num_gpus_per_ens=getattr(self.config.hardware, "num_gpus_per_ensemble", 1),
            num_gpus_per_model=self.config.hardware.num_gpus_per_model,
        )

    def _get_dataloader(self, ds: NativeGridDataset, stage: str) -> DataLoader:
        assert stage in {"training", "validation", "test"}
        return DataLoader(
            ds,
            batch_size=self.config.dataloader.batch_size[stage],
            # number of worker processes
            num_workers=self.config.dataloader.num_workers[stage],
            # use of pinned memory can speed up CPU-to-GPU data transfers
            # see https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning
            pin_memory=self.config.dataloader.pin_memory,
            # worker initializer
            worker_init_fn=worker_init_func,
            # prefetch batches
            prefetch_factor=self.config.dataloader.prefetch_factor,
            persistent_workers=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_train, "training")

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_valid, "validation")

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_test, "test")
