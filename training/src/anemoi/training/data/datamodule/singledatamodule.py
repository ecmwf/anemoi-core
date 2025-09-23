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

from anemoi.datasets.data import open_dataset
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.data.dataset.multidataset import MultiDataset
from anemoi.training.data.grid_indices import BaseGridIndices
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.utils.worker_init import worker_init_func
from anemoi.utils.dates import frequency_to_seconds

LOGGER = logging.getLogger(__name__)


class AnemoiDatasetsDataModule(pl.LightningDataModule):
    """Anemoi Datasets data module for PyTorch Lightning."""

    def __init__(self, config: BaseSchema, graph_data: HeteroData) -> None:
        """Initialize Multi-dataset data module.

        Parameters
        ----------
        config : BaseSchema
            Job configuration with multi-dataset specification
        graph_data : HeteroData
            Graph data for the model
        """
        super().__init__()

        self.config = config
        self.graph_data = graph_data

        # Validate that we have multiple datasets defined
        if not hasattr(self.config.dataloader.training, "datasets"):
            msg = (
                "Multi-dataset configuration requires 'datasets' section in training config. "
                "Use datasets: {name_a: {dataset: path_a, ...}, name_b: {dataset: path_b, ...}}"
            )
            raise ValueError(msg)

        self.dataset_names = list(self.config.dataloader.training.datasets.keys())
        LOGGER.info("Initializing multi-dataset module with datasets: %s", self.dataset_names)

        # Set training end dates if not specified for each dataset
        for name, dataset_config in self.config.dataloader.training.datasets.items():
            if dataset_config.end is None:
                LOGGER.info(
                    "No end date specified for training dataset '%s', setting default before validation start date %s.",
                    name,
                    self.config.dataloader.validation.start - 1,
                )
                dataset_config.end = self.config.dataloader.validation.start - 1

        if not self.config.dataloader.pin_memory:
            LOGGER.info("Data loader memory pinning disabled.")

    @cached_property
    def statistics(self) -> dict:
        """Return statistics from all training datasets."""
        return self.ds_train.statistics

    @cached_property
    def statistics_tendencies(self) -> dict:
        """Return tendency statistics from all training datasets."""
        return self.ds_train.statistics_tendencies

    @cached_property
    def metadata(self) -> dict:
        """Return metadata from all training datasets."""
        return self.ds_train.metadata

    @cached_property
    def supporting_arrays(self) -> dict:
        """Return supporting arrays from all training datasets."""
        # Each dataset has its own supporting arrays, no assumptions about sharing
        return self.ds_train.supporting_arrays

    @cached_property
    def data_indices(self) -> dict[str, IndexCollection]:
        """Return data indices for each dataset."""
        from anemoi.training.utils.config_utils import get_dataset_data_config

        indices = {}
        for dataset_name in self.dataset_names:
            name_to_index = self.ds_train.name_to_index[dataset_name]
            # Get dataset-specific data config
            data_config = get_dataset_data_config(self.config, dataset_name)
            indices[dataset_name] = IndexCollection(data_config, name_to_index)
        return indices

    def relative_date_indices(self, val_rollout: int = 1) -> list:
        """Determine a list of relative time indices to load for each batch."""
        if hasattr(self.config.training, "explicit_times"):
            return sorted(set(self.config.training.explicit_times.input + self.config.training.explicit_times.target))

        # Calculate indices using multistep, timeincrement and rollout
        rollout_cfg = getattr(getattr(self.config, "training", None), "rollout", None)

        rollout_max = getattr(rollout_cfg, "max", None)
        rollout_start = getattr(rollout_cfg, "start", 1)
        rollout_epoch_increment = getattr(rollout_cfg, "epoch_increment", 0)

        rollout_value = rollout_start
        if rollout_cfg and rollout_epoch_increment > 0 and rollout_max is not None:
            rollout_value = rollout_max
        else:
            LOGGER.warning("Falling back rollout to: %s", rollout_value)

        rollout = max(rollout_value, val_rollout)
        multi_step = self.config.training.multistep_input
        return [self.timeincrement * mstep for mstep in range(multi_step + rollout)]

    def add_trajectory_ids(self, data_reader: Callable) -> Callable:
        """Add trajectory IDs to data reader for forecast trajectory tracking."""
        if not hasattr(self.config.dataloader, "model_run_info"):
            data_reader.trajectory_ids = None
            return data_reader

        mr_start = np.datetime64(self.config.dataloader.model_run_info.start)
        mr_len = self.config.dataloader.model_run_info.length

        if hasattr(self.config.training, "rollout") and self.config.training.rollout.max is not None:
            max_rollout_index = max(self.relative_date_indices(self.config.training.rollout.max))
            assert (
                max_rollout_index < mr_len
            ), f"Requested data length {max_rollout_index + 1} longer than model run length {mr_len}"

        data_reader.trajectory_ids = (data_reader.dates - mr_start) // np.timedelta64(
            mr_len * frequency_to_seconds(self.config.data.frequency),
            "s",
        )
        return data_reader

    @cached_property
    def grid_indices(self) -> dict[str, type[BaseGridIndices]]:
        """Initialize grid indices for spatial sharding for each dataset."""
        reader_group_size = self.config.dataloader.read_group_size

        grid_indices_dict = {}

        # Each dataset can have its own grid indices configuration
        for dataset_name in self.dataset_names:
            if dataset_name in self.config.dataloader.grid_indices_per_dataset:
                grid_config = self.config.dataloader.grid_indices_per_dataset[dataset_name]
            else:
                # Fallback to default grid_indices config
                grid_config = self.config.dataloader.grid_indices

            grid_indices = instantiate(grid_config, reader_group_size=reader_group_size)
            grid_indices.setup(self.graph_data[dataset_name])
            grid_indices_dict[dataset_name] = grid_indices

        return grid_indices_dict

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
    def ds_train(self) -> MultiDataset:
        """Create multi-dataset for training."""
        datasets_config = {}
        for name, dataset_config in self.config.dataloader.training.datasets.items():
            data_reader = open_dataset(dataset_config)
            data_reader = self.add_trajectory_ids(data_reader)
            datasets_config[name] = data_reader

        return MultiDataset(
            datasets_config=datasets_config,
            grid_indices_config=self.grid_indices,
            relative_date_indices=self.relative_date_indices(),
            timestep=self.config.data.timestep,
            shuffle=True,
            label="train",
        )

    @cached_property
    def ds_valid(self) -> MultiDataset:
        """Create multi-dataset for validation."""
        datasets_config = {}
        for name, dataset_config in self.config.dataloader.validation.datasets.items():
            data_reader = open_dataset(dataset_config)
            data_reader = self.add_trajectory_ids(data_reader)
            datasets_config[name] = data_reader

        return MultiDataset(
            datasets_config=datasets_config,
            grid_indices_config=self.grid_indices,
            relative_date_indices=self.relative_date_indices(self.config.dataloader.validation_rollout),
            timestep=self.config.data.timestep,
            shuffle=False,
            label="validation",
        )

    @cached_property
    def ds_test(self) -> MultiDataset:
        """Create multi-dataset for testing."""
        datasets_config = {}
        for name, dataset_config in self.config.dataloader.test.datasets.items():
            data_reader = open_dataset(dataset_config)
            data_reader = self.add_trajectory_ids(data_reader)
            datasets_config[name] = data_reader

        return MultiDataset(
            datasets_config=datasets_config,
            grid_indices_config=self.grid_indices,
            relative_date_indices=self.relative_date_indices(),
            timestep=self.config.data.timestep,
            shuffle=False,
            label="test",
        )

    def _get_dataloader(self, ds: MultiDataset, stage: str) -> DataLoader:
        """Create DataLoader for multi-dataset."""
        assert stage in {"training", "validation", "test"}
        return DataLoader(
            ds,
            batch_size=self.config.dataloader.batch_size[stage],
            num_workers=self.config.dataloader.num_workers[stage],
            pin_memory=self.config.dataloader.pin_memory,
            worker_init_fn=worker_init_func,
            prefetch_factor=self.config.dataloader.prefetch_factor,
            persistent_workers=True,
        )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return self._get_dataloader(self.ds_train, "training")

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return self._get_dataloader(self.ds_valid, "validation")

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return self._get_dataloader(self.ds_test, "test")