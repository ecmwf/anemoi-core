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

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.training.data.data_reader import create_dataset
from anemoi.training.data.multidataset import MultiDataset
from anemoi.training.data.relative_time_indices import compute_model_relative_date_indices
from anemoi.training.data.relative_time_indices import compute_relative_date_indices
from anemoi.training.data.relative_time_indices import parse_dataset_time_indices_config
from anemoi.training.data.relative_time_indices import resolve_relative_date_indices
from anemoi.training.utils.worker_init import worker_init_func
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

if TYPE_CHECKING:
    from anemoi.training.schemas.base_schema import BaseSchema
    from anemoi.training.tasks.base import BaseTask

LOGGER = logging.getLogger(__name__)


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

    @cached_property
    def statistics(self) -> dict:
        """Return statistics from all training datasets."""
        return self.ds_train.statistics

    @cached_property
    def statistics_tendencies(self) -> dict[str, dict | None] | None:
        """Return tendency statistics from all training datasets."""
        lead_times = [frequency_to_string(offset) for offset in self.task.get_output_offsets()]

        stats_by_dataset: dict[str, dict | None] = {}
        for dataset_name, dataset in self.ds_train.datasets.items():
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

    def _lead_time_for_step(self, step: int) -> str:
        timestep = frequency_to_timedelta(self.config.data.timestep)
        return frequency_to_string(timestep * step)

    def relative_date_indices(self, mode: str = "training") -> list:
        """Determine a list of relative time indices to load for each batch."""
        return resolve_relative_date_indices(self.config, task=self.task, mode=mode, logger=LOGGER)

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

    def _get_dataset(
        self,
        config: dict[str, dict],
        shuffle: bool = True,
        label: str = "generic",
    ) -> MultiDataset:
        debug_cfg = getattr(self.config.dataloader, "debug", {})
        time_index_mode = getattr(debug_cfg, "time_index_mode", "dense")
        time_index_anchor_dataset = getattr(debug_cfg, "time_index_anchor_dataset", None)
        dataset_time_indices = parse_dataset_time_indices_config(self.config)
        data_readers = {name: create_dataset(data_reader, task=self.task) for name, data_reader in config.items()}
        model_relative_date_indices = compute_model_relative_date_indices(self.task, mode=label)
        relative_date_indices: list | dict[str, list[int]]
        if model_relative_date_indices is not None:
            relative_date_indices = model_relative_date_indices
        else:
            relative_date_indices = compute_relative_date_indices(self.task, data_readers, mode=label)
        return MultiDataset(
            data_readers=data_readers,
            relative_date_indices=relative_date_indices,
            timestep=self.config.data.timestep,
            multistep_window=getattr(self.config.training, "multistep_window", None),
            explicit_time_indices_by_dataset=dataset_time_indices,
            time_index_mode=time_index_mode,
            time_index_anchor_dataset=time_index_anchor_dataset,
            shuffle=shuffle,
            label=label,
            debug=debug_cfg,
        )

    def _get_dataloader(self, ds: MultiDataset, stage: str) -> DataLoader:
        """Create DataLoader for multi-dataset."""
        assert stage in {"training", "validation", "test"}
        num_workers = int(self.config.dataloader.num_workers[stage])
        prefetch_factor = self.config.dataloader.prefetch_factor if num_workers > 0 else None
        persistent_workers = num_workers > 0
        return DataLoader(
            ds,
            batch_size=self.config.dataloader.batch_size[stage],
            num_workers=num_workers,
            pin_memory=self.config.dataloader.pin_memory,
            worker_init_fn=worker_init_func,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
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

        def _dataset_timing_metadata(dataset: MultiDataset, label: str) -> dict[str, list[int] | dict[str, list[int]]]:
            return {
                f"relative_date_indices_{label}": [int(v) for v in dataset.model_relative_date_indices.tolist()],
                f"relative_date_input_indices_{label}_by_dataset": {
                    name: [int(v) for v in dataset.input_model_relative_date_indices_by_dataset[name].tolist()]
                    for name in self.dataset_names
                },
                f"relative_date_input_indices_{label}_native_by_dataset": {
                    name: [int(v) for v in dataset.input_data_relative_date_indices_by_dataset[name].tolist()]
                    for name in self.dataset_names
                },
                f"relative_date_indices_{label}_by_dataset": {
                    name: [int(v) for v in dataset.model_relative_date_indices_by_dataset[name].tolist()]
                    for name in self.dataset_names
                },
                f"relative_date_indices_{label}_native_by_dataset": {
                    name: [int(v) for v in dataset.data_relative_date_indices_by_dataset[name].tolist()]
                    for name in self.dataset_names
                },
                f"relative_date_target_indices_{label}_by_dataset": {
                    name: [int(v) for v in dataset.target_model_relative_date_indices_by_dataset[name].tolist()]
                    for name in self.dataset_names
                },
                f"relative_date_target_indices_{label}_native_by_dataset": {
                    name: [int(v) for v in dataset.target_data_relative_date_indices_by_dataset[name].tolist()]
                    for name in self.dataset_names
                },
            }

        timesteps = {
            **_dataset_timing_metadata(self.ds_train, "training"),
            "timestep": self.config.data.timestep,
        }
        if len(self.valid_dataloader_config) > 0:
            timesteps.update(_dataset_timing_metadata(self.ds_valid, "validation"))

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
