# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from functools import cache, cached_property

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.training.data.multidataset import MultiDataset
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.utils.worker_init import worker_init_func

LOGGER = logging.getLogger(__name__)


class AnemoiDatasetsDataModule(pl.LightningDataModule):
    """Anemoi Datasets data module for PyTorch Lightning."""

    def __init__(self, config: BaseSchema, task: "BaseTask") -> None:
        """Initialize Multi-dataset data module.

        Parameters
        ----------
        config : BaseSchema
            Job configuration with multi-dataset specification
        task : BaseTask
            Task defining the problem to solve
        """
        super().__init__()

        self.config = config
        self.task = task

        self.train_dataloader_config = get_multiple_datasets_config(self.config.dataloader.training)
        self.valid_dataloader_config = get_multiple_datasets_config(self.config.dataloader.validation)
        self.test_dataloader_config = get_multiple_datasets_config(self.config.dataloader.test)

        self.dataset_names = list(self.train_dataloader_config.keys())
        LOGGER.info("Initializing multi-dataset module with datasets: %s", self.dataset_names)

        # Set training end dates if not specified for each dataset
        for name, dataset_config in self.train_dataloader_config.items():
            if dataset_config.end is None:
                msg = f"No end date specified for training dataset {name}."
                raise ValueError(msg)

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
        return self.ds_train.supporting_arrays

    @cached_property
    def data_indices(self) -> dict[str, IndexCollection]:
        """Return data indices for each dataset."""
        indices = {}
        data_config = get_multiple_datasets_config(self.config.data)
        for dataset_name in self.dataset_names:
            name_to_index = self.ds_train.name_to_index[dataset_name]
            # Get dataset-specific data config
            indices[dataset_name] = IndexCollection(data_config[dataset_name], name_to_index)
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
        return list(range(multi_step + rollout))

    @cached_property
    def ds_train(self) -> MultiDataset:
        """Create multi-dataset for training."""
        return self._get_dataset(
            self.train_dataloader_config,
            shuffle=True,
            label="training",
        )

    @cached_property
    def ds_valid(self) -> MultiDataset:
        """Create multi-dataset for validation."""
        return self._get_dataset(
            self.valid_dataloader_config,
            shuffle=False,
            val_rollout=self.config.dataloader.validation_rollout,
            label="validation",
        )

    @cached_property
    def ds_test(self) -> MultiDataset:
        """Create multi-dataset for testing."""
        return self._get_dataset(self.test_dataloader_config, shuffle=False, label="test")

    def _get_dataset(
        self,
        datasets: dict[str, dict],
        shuffle: bool = True,
        val_rollout: int = 0,
        label: str = "generic",
    ) -> MultiDataset:
        relative_date_indices = {
            dataset_name: self.task.get_relative_time_indices(ds.frequency) for dataset_name, ds in datasets.items()
        }

        return MultiDataset(
            data_readers=datasets,
            relative_date_indices=relative_date_indices,
            timestep=self.config.data.timestep,
            shuffle=shuffle,
            label=label,
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

    def fill_metadata(self, metadata: dict) -> None:
        """Fill metadata dictionary with dataset metadata."""
        datasets_config = self.metadata.copy()
        metadata["dataset"] = datasets_config
        data_indices = self.data_indices.copy()
        metadata["data_indices"] = data_indices

        metadata["metadata_inference"]["dataset_names"] = self.dataset_names

        timesteps = {
            #"relative_date_indices_training": self.relative_date_indices(),
            "timestep": self.config.data.timestep,
        }
        for dataset_name in self.dataset_names:
            metadata["metadata_inference"][dataset_name] = {}
            metadata["metadata_inference"][dataset_name]["timesteps"] = timesteps

            name_to_index = {
                "input": data_indices[dataset_name].model.input.name_to_index,
                "output": data_indices[dataset_name].model.output.name_to_index,
            }
            metadata["metadata_inference"][dataset_name]["data_indices"] = name_to_index

            input_data_indices = data_indices[dataset_name].data.input.todict()
            input_index_to_name = {v: k for k, v in input_data_indices["name_to_index"].items()}
            variable_types = {
                "forcing": [input_index_to_name[int(index)] for index in input_data_indices["forcing"]],
                "target": [input_index_to_name[int(index)] for index in input_data_indices["target"]],
                "prognostic": [input_index_to_name[int(index)] for index in input_data_indices["prognostic"]],
                "diagnostic": [input_index_to_name[int(index)] for index in input_data_indices["diagnostic"]],
            }
            metadata["metadata_inference"][dataset_name]["variable_types"] = variable_types
