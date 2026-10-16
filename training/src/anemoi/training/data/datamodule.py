# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
from functools import cached_property
from typing import Any

import pytorch_lightning as pl
from hydra.utils import instantiate
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import BatchSampler

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.training.data.data_reader import create_dataset
from anemoi.training.data.loading import DataLoadingPolicy
from anemoi.training.data.multidataset import MultiDataset
from anemoi.training.data.relative_time_indices import resolve_task_relative_indices_by_dataset
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.tasks.base import BaseTask
from anemoi.training.utils.seeding import SeedContext
from anemoi.training.utils.seeding import derive_seed
from anemoi.training.utils.seeding import get_base_seed
from anemoi.utils.dates import frequency_to_string

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

        self.epoch = 0

    @cached_property
    def statistics(self) -> dict:
        """Return statistics from all training datasets."""
        return self.ds_train.statistics

    @cached_property
    def statistics_tendencies(self) -> dict[str, dict | None] | None:
        """Return tendency statistics from all training datasets."""
        stats_by_dataset: dict[str, dict | None] = {}
        for dataset_name, dataset in self.ds_train.data_readers.items():
            targets_by_step = self.task.dataset_target_relative_times_by_dataset_by_step.get(dataset_name)
            if targets_by_step is not None and self.task.model_timestep is not None:
                lead_times = [
                    frequency_to_string(relative_time * self.task.model_timestep)
                    for relative_time in targets_by_step[0]
                ]
            else:
                lead_times = [frequency_to_string(step) for step in self.task.get_output_offsets()]
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
    def loading_policy(self) -> DataLoadingPolicy:
        """Return the policy that builds the dataset and batch sampler for each stage."""
        return instantiate(self.config.dataloader.loading_policy)

    @cached_property
    def ds_train(self) -> MultiDataset:
        """Create multi-dataset for training."""
        return self._get_dataset(self.train_dataloader_config, label="training")

    @cached_property
    def ds_valid(self) -> MultiDataset:
        """Create multi-dataset for validation."""
        return self._get_dataset(self.valid_dataloader_config, label="validation")

    @cached_property
    def ds_test(self) -> MultiDataset:
        """Create multi-dataset for testing."""
        return self._get_dataset(self.test_dataloader_config, label="test")

    def _get_dataset(
        self,
        config: dict[str, dict],
        label: str = "generic",
    ) -> MultiDataset:
        data_readers = {name: create_dataset(data_reader, task=self.task) for name, data_reader in config.items()}
        dataset_options = {}
        dataloader_config = getattr(getattr(self, "config", None), "dataloader", {})
        if dataloader_config.get("fake_dataloading", False):
            dataset_options["fake_dataloading"] = True

        relative_date_indices = self.loading_policy.relative_date_indices(
            self.task,
            data_readers,
            mode=label,
        )
        return self.loading_policy.build_dataset(
            data_readers,
            relative_date_indices,
            **dataset_options,
        )

    def set_epoch(self, epoch: int) -> None:
        """Set the datamodule epoch and synchronize datasets settings."""
        self.epoch = epoch
        self.sync_dataset_state()

    def sync_dataset_state(self) -> None:
        """Load the time steps that the task's current rollout needs in all constructed datasets."""
        refresh_task_timing = False
        for dataset_name, label in (("ds_train", "training"), ("ds_valid", "validation"), ("ds_test", "test")):
            if dataset_name not in self.__dict__:
                continue

            dataset = self.__dict__[dataset_name]
            relative_date_indices = self.loading_policy.relative_date_indices(
                self.task,
                dataset.data_readers,
                mode=label,
            )
            dataset.set_relative_date_indices(
                relative_date_indices=relative_date_indices,
            )
            refresh_task_timing |= not dataset.relative_date_indices_are_native

        if refresh_task_timing:
            timesteps = self._timing_metadata()
            metadata = {
                "metadata_inference": {
                    "dataset_names": self.dataset_names,
                    **{dataset_name: {"timesteps": timesteps} for dataset_name in self.dataset_names},
                },
            }
            self.task.fill_metadata(metadata)
            self.task.configure_from_metadata(metadata)

    def state_dict(self) -> dict[str, Any]:
        """Save the epoch that selects the training shuffle of newly built dataloaders."""
        return {"epoch": self.epoch}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the epoch before Lightning builds the dataloaders."""
        self.set_epoch(state_dict["epoch"])

    @cached_property
    def rollout_changes_between_epochs(self) -> bool:
        """Whether the rollout, and with it the number of samples, changes between epochs.

        The dataloaders must then be rebuilt every epoch, because a sampler reads
        the number of samples when it is constructed.
        """
        rollout = getattr(self.task, "rollout", None)
        return rollout is not None and rollout.epoch_increment > 0

    def _sample_groups(self) -> dict[str, int]:
        """Return the number of sample groups and the group of this rank.

        All ranks of a group train on the same samples. Without a distributed
        strategy, or without a trainer, there is a single group.
        """
        trainer = self.trainer
        if trainer is None or trainer.distributed_sampler_kwargs is None:
            return {"num_replicas": 1, "rank": 0}
        return trainer.distributed_sampler_kwargs

    def _get_batch_sampler(self, ds: MultiDataset, stage: str) -> BatchSampler:
        """Create the batch sampler that splits the samples between sample groups.

        Every group gets the same number of samples. The training shuffle depends
        only on the base seed and the epoch. The batch sampler counts the samples it
        has handed out; Lightning saves this count, which is the same on every rank,
        so training can resume in the middle of an epoch.
        """
        batch_sampler = self.loading_policy.build_batch_sampler(
            ds,
            **self._sample_groups(),
            shuffle=(stage == "training"),
            seed=derive_seed(get_base_seed(), SeedContext.DATALOADER),
            batch_size=self.config.dataloader.batch_size[stage],
            drop_last=True,
        )
        # Lightning sets the epoch when an epoch starts, but a resumed run builds
        # its first dataloader before that, so the restored epoch is set here.
        batch_sampler.sampler.set_epoch(self.epoch)
        return batch_sampler

    def _get_dataloader(self, ds: MultiDataset, stage: str) -> StatefulDataLoader:
        """Create DataLoader for multi-dataset."""
        assert stage in {"training", "validation", "test"}

        extra = {}

        if self.config.dataloader.get("multiprocessing_context", None) is not None:
            import multiprocessing

            ctx = self.config.dataloader.multiprocessing_context
            extra["multiprocessing_context"] = multiprocessing.get_context(ctx)

            LOGGER.info("Using multiprocessing context '%s' for dataloader workers.", ctx)

        return StatefulDataLoader(
            ds,
            batch_sampler=self._get_batch_sampler(ds, stage),
            num_workers=self.config.dataloader.num_workers[stage],
            pin_memory=self.config.dataloader.pin_memory,
            prefetch_factor=self.config.dataloader.prefetch_factor,
            persistent_workers=self.config.dataloader.get("persistent_workers", False),
            **extra,
        )

    def train_dataloader(self) -> StatefulDataLoader:
        """Return training dataloader."""
        return self._get_dataloader(self.ds_train, "training")

    def val_dataloader(self) -> StatefulDataLoader:
        """Return validation dataloader."""
        return self._get_dataloader(self.ds_valid, "validation")

    def test_dataloader(self) -> StatefulDataLoader:
        """Return test dataloader."""
        return self._get_dataloader(self.ds_test, "test")

    def fill_metadata(self, metadata: dict) -> None:
        """Fill metadata dictionary with dataset metadata."""
        datasets_config = self.metadata.copy()
        metadata["dataset"] = datasets_config
        data_indices = self.data_indices.copy()
        metadata["data_indices"] = data_indices

        metadata["metadata_inference"]["dataset_names"] = self.dataset_names

        timesteps = self._timing_metadata()

        for dataset_name in self.dataset_names:
            metadata["metadata_inference"][dataset_name] = {}
            if len(timesteps) > 0:
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

    def _timing_metadata(self) -> dict[str, object]:
        """Return the current dataset-specific time windows for task configuration and inference metadata."""
        decoders = getattr(getattr(self.config, "model", None), "decoders", None)
        target_dataset_names = (
            {dataset_name for decoder in decoders.values() for dataset_name in decoder.target_datasets}
            if decoders is not None
            else None
        )

        def _dataset_timing_metadata(dataset: MultiDataset, label: str) -> dict[str, object]:
            if dataset.relative_date_indices_are_native:
                return {}
            assert dataset.frequency_seconds is not None

            (
                input_relative_indices_by_dataset,
                target_relative_indices_by_dataset,
                target_relative_indices_by_dataset_by_step,
            ) = resolve_task_relative_indices_by_dataset(
                self.task,
                dataset.model_relative_date_indices_by_dataset,
                datetime.timedelta(seconds=dataset.frequency_seconds),
                mode=label,
            )
            if target_dataset_names is not None:
                for dataset_name in set(self.dataset_names) - target_dataset_names:
                    target_relative_indices_by_dataset[dataset_name] = []
                    target_relative_indices_by_dataset_by_step[dataset_name] = [[] for _step in self.task.steps(label)]

            return {
                "model_timestep": frequency_to_string(datetime.timedelta(seconds=dataset.frequency_seconds)),
                f"relative_date_indices_{label}": [int(v) for v in dataset.model_relative_date_indices.tolist()],
                f"relative_date_input_indices_{label}_by_dataset": input_relative_indices_by_dataset,
                f"relative_date_indices_{label}_by_dataset": {
                    name: [int(v) for v in dataset.model_relative_date_indices_by_dataset[name].tolist()]
                    for name in self.dataset_names
                },
                f"relative_date_target_indices_{label}_by_dataset": target_relative_indices_by_dataset,
                f"relative_date_target_indices_{label}_by_dataset_by_step": (
                    target_relative_indices_by_dataset_by_step
                ),
            }

        timesteps = _dataset_timing_metadata(self.ds_train, "training")
        if len(self.valid_dataloader_config) > 0:
            timesteps.update(_dataset_timing_metadata(self.ds_valid, "validation"))
        return timesteps
