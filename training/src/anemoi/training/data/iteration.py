# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import itertools
import logging
import os
from collections.abc import Generator
from typing import TYPE_CHECKING

import numpy as np
import torch

from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.training.data.usable_indices import compute_valid_anchors
from anemoi.training.utils.time_indices import TimeIndices
from anemoi.training.utils.time_indices import offset_time_indices

if TYPE_CHECKING:
    from anemoi.training.data.multidataset import MultiDataset

LOGGER = logging.getLogger(__name__)


class BaseIteration:
    """Sample synchronized data from all readers of a worker dataset."""

    def compute_anchors(
        self,
        dataset: "MultiDataset",
        relative_date_indices: dict[str, TimeIndices],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute synchronized anchors shared by all data readers."""
        anchors = compute_valid_anchors(dataset.data_readers, relative_date_indices)
        return anchors, np.arange(len(anchors), dtype=np.int64)

    def _grid_indices(self, dataset: "MultiDataset", dataset_name: str) -> slice:
        # dataset.shard_sizes is lazily initalised to None
        # This if statement guards against the case where shard_sizes is not set
        # (e.g. if set_comm_group_info hasn't been called yet)
        if dataset.shard_sizes is not None and dataset.shard_sizes[dataset_name] is not None:
            start, end = get_partition_range(
                dataset.shard_sizes[dataset_name],
                dataset.reader_group_rank,
            )
            return slice(start, end)
        return slice(None)

    def sample(self, dataset: "MultiDataset", index: int) -> dict[str, torch.Tensor]:
        """Load one synchronized sample from every reader."""
        sequence, position = (int(value) for value in dataset.anchors[index])
        return {
            name: reader.get_sample(
                sequence,
                offset_time_indices(position, dataset.relative_date_indices[name]),
                self._grid_indices(dataset, name),
            )
            for name, reader in dataset.data_readers.items()
        }

    def _sample_indices(self, dataset: "MultiDataset") -> np.ndarray:
        # All data readers use the same shuffled anchor indices for synchronization.
        if dataset.shuffle:
            indices = dataset.rng.choice(
                dataset.valid_date_indices,
                size=len(dataset.valid_date_indices),
                replace=False,
            )[dataset.chunk_index_range]
        else:
            indices = dataset.valid_date_indices[dataset.chunk_index_range]

        LOGGER.debug(
            "%s worker pid %d, worker id %d, using synchronized indices[0:10]: %s",
            dataset.__class__.__name__,
            os.getpid(),
            dataset.worker_id,
            indices[:10],
        )
        return indices

    def __call__(self, dataset: "MultiDataset") -> Generator[dict[str, torch.Tensor], None, None]:
        """Yield the samples assigned to a worker dataset."""
        initial_batch = None
        for index in self._sample_indices(dataset):
            if not dataset.fake_dataloading:
                yield self.sample(dataset, index)
            elif initial_batch is None:
                initial_batch = self.sample(dataset, index)
                yield initial_batch
            else:
                yield initial_batch


class CrossDatasetIteration(BaseIteration):
    """Sample one independently indexed dataset at a time."""

    def compute_anchors(
        self,
        dataset: "MultiDataset",
        relative_date_indices: dict[str, TimeIndices],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Compute independent anchors for each dataset."""
        anchors = {
            name: data_reader.compute_anchors(relative_date_indices[name])
            for name, data_reader in dataset.data_readers.items()
        }
        for name, values in anchors.items():
            if len(values) == 0:
                msg = f"No valid anchors found for data reader '{name}': {dataset.data_readers[name]}"
                raise ValueError(msg)
        return anchors, {name: np.arange(len(values), dtype=np.int64) for name, values in anchors.items()}

    def sample(self, dataset: "MultiDataset", index: tuple[str, int]) -> dict[str, torch.Tensor]:
        """Load one sample from the selected dataset."""
        dataset_name, sample_index = index
        sequence, position = (int(value) for value in dataset.anchors[dataset_name][sample_index])
        time_steps = offset_time_indices(position, dataset.relative_date_indices[dataset_name])
        return {
            dataset_name: dataset.data_readers[dataset_name].get_sample(
                sequence,
                time_steps,
                self._grid_indices(dataset, dataset_name),
            ),
        }

    def _sample_indices(self, dataset: "MultiDataset") -> list[tuple[str, int]]:
        dataset_indices = {
            name: (
                dataset.rng.choice(indices, size=len(indices), replace=False)[dataset.chunk_index_range[name]]
                if dataset.shuffle
                else indices[dataset.chunk_index_range[name]]
            )
            for name, indices in dataset.valid_date_indices.items()
        }
        samples = [(name, int(index)) for name, indices in dataset_indices.items() for index in indices]
        if dataset.shuffle and len(dataset_indices) > 1:
            order = dataset.rng.choice(len(samples), size=len(samples), replace=False)
            samples = [samples[int(index)] for index in order]

        LOGGER.debug(
            (
                "Worker pid %d, label %s, worker id %d, global_rank %d, "
                "model comm group %d, group_rank %d, seed comm group id %d, using indices[0:10]: %s"
            ),
            os.getpid(),
            dataset.label,
            dataset.worker_id,
            dataset.global_rank,
            dataset.model_comm_group_id,
            dataset.model_comm_group_rank,
            dataset.sample_comm_group_id,
            samples[:10],
        )
        return samples


class MixedIteration(BaseIteration):
    """Mixing cross-dataset iteration logic and base iteration logic."""

    def __init__(self, iteration_mapping: list):
        self.iteration_mapping = iteration_mapping

    def compute_anchors(
        self,
        dataset: "MultiDataset",
        relative_date_indices: dict[str, TimeIndices],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:

        # Create groups of datasets that will be sampled together
        groups = list(itertools.product(*self.iteration_mapping))
        self.groups_dict = {"group_" + str(i): group for i, group in enumerate(groups)}
        self.group_labels = list(self.groups_dict.keys())

        anchors = {}
        for group, datasets_in_group in self.groups_dict.items():
            data_readers_in_group = {}
            for dataset_label in datasets_in_group:
                if dataset_label in dataset.data_readers:
                    data_readers_in_group[dataset_label] = dataset.data_readers[dataset_label]
                else:
                    msg = f"Dataset label '{dataset_label}' not found in data readers for group '{group}'."
                    LOGGER.warning(msg)
                    continue

            if not data_readers_in_group:
                msg = f"Group '{group}' has no valid data readers. Skipping this group."
                LOGGER.warning(msg)
                continue

            LOGGER.debug("Data readers in group %s: %s", group, list(data_readers_in_group.keys()))
            group_anchors = compute_valid_anchors(
                data_readers_in_group,
                relative_date_indices,
            )
            if group_anchors is not None and len(group_anchors) > 0:
                anchors[group] = group_anchors
                LOGGER.info("Group '%s' has %d valid anchors", group, len(group_anchors))
            else:
                LOGGER.warning("Group '%s' has no valid anchors", group)

        if not anchors:
            available_datasets = list(dataset.data_readers.keys())
            msg = (
                f"No valid anchors found after intersection across all dataset groups. "
                f"Available datasets: {available_datasets}, Requested groups: {self.groups_dict}"
            )
            raise ValueError(msg)

        LOGGER.info("Computed anchors for groups: %s", {k: len(v) for k, v in anchors.items()})
        return anchors, {group_name: np.arange(len(values), dtype=np.int64) for group_name, values in anchors.items()}

    def sample(
        self,
        dataset: "MultiDataset",
        index: tuple[str, int],
    ) -> dict[str, torch.Tensor]:
        """Load one sample from all datasets in the selected group."""
        group_name, sample_index = index

        sequence, position = (int(value) for value in dataset.anchors[group_name][sample_index])

        result = {}
        for dataset_name in self.groups_dict[group_name]:
            if dataset_name not in dataset.data_readers:
                LOGGER.warning("Dataset '%s' not found in data_readers for group '%s'", dataset_name, group_name)
                continue

            sample = dataset.data_readers[dataset_name].get_sample(
                sequence,
                offset_time_indices(position, dataset.relative_date_indices[dataset_name]),
                self._grid_indices(dataset, dataset_name),
            )
            result[dataset_name] = sample

        if not result:
            LOGGER.error(
                "No samples were retrieved for group '%s'. Available data readers: %s, Expected datasets: %s",
                group_name,
                list(dataset.data_readers.keys()),
                self.groups_dict[group_name],
            )
        LOGGER.info(
            "Sampled from group '%s' with datasets: %s, sample index: %d, data shapes: %s",
            group_name,
            list(result.keys()),
            sample_index,
            {k: v.shape for k, v in result.items()},
        )

        return result

    def _sample_indices(self, dataset: "MultiDataset") -> list[tuple[str, int]]:
        dataset_indices = {
            group_name: (
                dataset.rng.choice(indices, size=len(indices), replace=False)[dataset.chunk_index_range[group_name]]
                if dataset.shuffle
                else indices[dataset.chunk_index_range[group_name]]
            )
            for group_name, indices in dataset.valid_date_indices.items()
        }
        samples = [(group_name, int(index)) for group_name, indices in dataset_indices.items() for index in indices]

        if dataset.shuffle and len(dataset_indices) > 1:
            order = dataset.rng.choice(len(samples), size=len(samples), replace=False)
            samples = [samples[int(index)] for index in order]

        LOGGER.info(
            (
                "Worker pid %d, label %s, worker id %d, global_rank %d, "
                "model comm group %d, group_rank %d, seed comm group id %d, using indices[0:10]: %s"
            ),
            os.getpid(),
            dataset.label,
            dataset.worker_id,
            dataset.global_rank,
            dataset.model_comm_group_id,
            dataset.model_comm_group_rank,
            dataset.sample_comm_group_id,
            samples[:10],
        )
        LOGGER.info("Generated %d sample indices for mixed iteration", len(samples))
        return samples
