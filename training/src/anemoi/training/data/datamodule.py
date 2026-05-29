# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections import defaultdict
from collections.abc import Iterator
from functools import cached_property
from typing import Literal

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.training.data.data_reader import create_dataset
from anemoi.training.data.multidataset import MultiDataset
from anemoi.training.data.relative_time_indices import compute_relative_date_indices
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.tasks.base import BaseTask
from anemoi.training.utils.worker_init import worker_init_func
from anemoi.utils.dates import frequency_to_string

LOGGER = logging.getLogger(__name__)

_SHARED_LIST_MAX_SIZE = 1024


class SharedListProxy:
    """Variable-length shared-memory list backed by torch tensors.

    Uses POSIX shared memory directly — no server process, no IPC sockets — so it avoids
    the extra resident-set overhead and the fragility of a background Manager
    process being OOM-killed.

    Works with all DataLoader multiprocessing contexts (fork, spawn,
    forkserver) because PyTorch serialises shared-tensor handles rather than
    copying the underlying data.
    """

    def __init__(self, values: list[int], max_size: int = _SHARED_LIST_MAX_SIZE) -> None:
        self._data = torch.zeros(max_size, dtype=torch.long).share_memory_()
        self._len = torch.zeros(1, dtype=torch.long).share_memory_()
        self[:] = values

    def __setitem__(self, key: slice, values: list[int]) -> None:
        n = len(values)
        if n > len(self._data):
            msg = f"{self.__class__.__name__} capacity {len(self._data)} exceeded by {n} elements."
            raise ValueError(msg)
        self._data[:n] = torch.tensor(values, dtype=torch.long)
        self._len[0] = n

    def __getitem__(self, index: int) -> int:
        return int(self._data[index])

    def __len__(self) -> int:
        return int(self._len[0])

    def __iter__(self) -> Iterator[int]:
        return iter(self._data[: len(self)].tolist())


class AnemoiDatasetsDataModule(pl.LightningDataModule):
    """Anemoi Datasets data module for PyTorch Lightning."""

    def __init__(self, config: BaseSchema, task: BaseTask) -> None:
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

        self._relative_date_indices_values: dict[str, dict] = defaultdict(dict)
        self._data_readers: dict[str, dict] = {}

    @cached_property
    def statistics(self) -> dict:
        """Return statistics from all training datasets."""
        return self.ds_train.statistics

    @cached_property
    def statistics_tendencies(self) -> dict[str, dict | None] | None:
        """Return tendency statistics from all training datasets."""
        lead_times = [frequency_to_string(step) for step in self.task.get_output_offsets()]

        stats_by_dataset: dict[str, dict | None] = {}
        for dataset_name, dataset in self.ds_train.data_readers.items():
            stats_by_lead = {lead_time: dataset.statistics_tendencies(lead_time) for lead_time in lead_times}
            if all(stats is None for stats in stats_by_lead.values()):
                stats_by_dataset[dataset_name] = None
                continue
            stats_by_lead["lead_times"] = lead_times
            stats_by_dataset[dataset_name] = stats_by_lead

        if not any(stats is not None for stats in stats_by_dataset.values()):
            return None
        return stats_by_dataset

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

    @cached_property
    def ds_train(self) -> MultiDataset:
        """Create multi-dataset for training."""
        return self._get_dataset(self.train_dataloader_config, shuffle=True, label="training")

    @cached_property
    def ds_valid(self) -> MultiDataset:
        """Create multi-dataset for validation."""
        return self._get_dataset(self.valid_dataloader_config, shuffle=False, label="validation")

    @cached_property
    def ds_test(self) -> MultiDataset:
        """Create multi-dataset for testing."""
        return self._get_dataset(self.test_dataloader_config, shuffle=False, label="test")

    def _get_relative_date_indices(self, data_readers: dict, label: str) -> dict[str, list[int]]:
        """Compute relative date indices for each dataset based on the task and data readers."""
        relative_date_indices = compute_relative_date_indices(self.task, data_readers, mode=label)

        if label not in self._relative_date_indices_values:
            self._relative_date_indices_values[label] = {}

        for ds in relative_date_indices:
            if ds in self._relative_date_indices_values[label]:
                # Update in-place so all workers immediately see the new values via shared memory
                self._relative_date_indices_values[label][ds][:] = relative_date_indices[ds]
            else:
                self._relative_date_indices_values[label][ds] = SharedListProxy(relative_date_indices[ds])

        return self._relative_date_indices_values[label]

    def _get_dataset(
        self,
        config: dict[str, dict],
        shuffle: bool = True,
        label: str = "generic",
    ) -> MultiDataset:
        data_readers = {name: create_dataset(data_reader, task=self.task) for name, data_reader in config.items()}
        self._data_readers[label] = data_readers

        relative_date_indices = self._get_relative_date_indices(data_readers, label)

        return MultiDataset(
            data_readers=data_readers,
            relative_date_indices=relative_date_indices,
            shuffle=shuffle,
            label=label,
        )

    def recalculate_relative_date_indices(
        self,
        *,
        datasets: list[Literal["training", "validation", "test"]] | None = None,
    ) -> None:
        """Invalidate cached datasets, allows for recalculation of `relative_date_indices` during training."""
        if datasets is None:
            datasets = ["training", "validation", "test"]
        for ds in datasets:
            if ds not in self._data_readers:
                msg = f"No data readers found for dataset '{ds}' when recalculating relative date indices."
                raise ValueError(msg)
            self._get_relative_date_indices(self._data_readers[ds], ds)

    def _get_dataloader(self, ds: MultiDataset, stage: str) -> DataLoader:
        """Create DataLoader for multi-dataset."""
        assert stage in {"training", "validation", "test"}

        extra = {}

        if self.config.dataloader.get("multiprocessing_context", None) is not None:
            import multiprocessing

            ctx = self.config.dataloader.multiprocessing_context
            extra["multiprocessing_context"] = multiprocessing.get_context(ctx)

            LOGGER.info("Using multiprocessing context '%s' for dataloader workers.", ctx)

        return DataLoader(
            ds,
            batch_size=self.config.dataloader.batch_size[stage],
            num_workers=self.config.dataloader.num_workers[stage],
            pin_memory=self.config.dataloader.pin_memory,
            worker_init_fn=worker_init_func,
            prefetch_factor=self.config.dataloader.prefetch_factor,
            persistent_workers=True,
            **extra,
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

        for dataset_name in self.dataset_names:
            metadata["metadata_inference"][dataset_name] = {}

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
