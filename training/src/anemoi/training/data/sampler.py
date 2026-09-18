# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
from collections.abc import Generator
from typing import TYPE_CHECKING

import numpy as np
import torch

from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.training.utils.time_indices import offset_time_indices

if TYPE_CHECKING:
    from anemoi.training.data.multidataset import MultiDataset

LOGGER = logging.getLogger(__name__)


class BaseSampler:
    """Sample synchronized data from all readers of a worker dataset."""

    def __init__(self, dataset: "MultiDataset") -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        """Return the number of samples assigned to the worker."""
        return len(self.dataset.chunk_index_range)

    def _grid_indices(self, dataset_name: str) -> slice:
        if self.dataset.shard_sizes is not None and self.dataset.shard_sizes[dataset_name] is not None:
            start, end = get_partition_range(
                self.dataset.shard_sizes[dataset_name],
                self.dataset.reader_group_rank,
            )
            return slice(start, end)
        return slice(None)

    def sample(self, index: int) -> dict[str, torch.Tensor]:
        """Load one synchronized sample from every reader."""
        sequence, position = (int(value) for value in self.dataset.anchors[index])
        return {
            name: reader.get_sample(
                sequence,
                offset_time_indices(position, self.dataset.relative_date_indices[name]),
                self._grid_indices(name),
            )
            for name, reader in self.dataset.data_readers.items()
        }

    def _sample_indices(self) -> np.ndarray:
        if self.dataset.shuffle:
            indices = self.dataset.rng.choice(
                self.dataset.valid_date_indices,
                size=len(self.dataset.valid_date_indices),
                replace=False,
            )[self.dataset.chunk_index_range]
        else:
            indices = self.dataset.valid_date_indices[self.dataset.chunk_index_range]

        LOGGER.debug(
            "%s worker pid %d, worker id %d, using synchronized indices[0:10]: %s",
            self.dataset.__class__.__name__,
            os.getpid(),
            self.dataset.worker_id,
            indices[:10],
        )
        return indices

    def __iter__(self) -> Generator[dict[str, torch.Tensor], None, None]:
        initial_batch = None
        for index in self._sample_indices():
            if not self.dataset.fake_dataloading:
                yield self.sample(index)
            elif initial_batch is None:
                initial_batch = self.sample(index)
                yield initial_batch
            else:
                yield initial_batch


class CrossDatasetSampler(BaseSampler):
    """Sample one independently indexed dataset at a time."""

    def __len__(self) -> int:
        """Return the number of samples assigned to the worker."""
        return sum(len(indices) for indices in self.dataset.chunk_index_range.values())

    def sample(self, index: tuple[str, int]) -> dict[str, torch.Tensor]:
        """Load one sample from the selected dataset."""
        dataset_name, sample_index = index
        sequence, position = (int(value) for value in self.dataset.anchors[dataset_name][sample_index])
        time_steps = offset_time_indices(position, self.dataset.relative_date_indices[dataset_name])
        return {
            dataset_name: self.dataset.data_readers[dataset_name].get_sample(
                sequence,
                time_steps,
                self._grid_indices(dataset_name),
            ),
        }

    def _sample_indices(self) -> list[tuple[str, int]]:
        dataset_indices = {
            name: (
                self.dataset.rng.choice(indices, size=len(indices), replace=False)[self.dataset.chunk_index_range[name]]
                if self.dataset.shuffle
                else indices[self.dataset.chunk_index_range[name]]
            )
            for name, indices in self.dataset.valid_date_indices.items()
        }
        samples = [(name, int(index)) for name, indices in dataset_indices.items() for index in indices]
        if self.dataset.shuffle:
            order = self.dataset.rng.choice(len(samples), size=len(samples), replace=False)
            samples = [samples[int(index)] for index in order]

        LOGGER.debug(
            (
                "Worker pid %d, label %s, worker id %d, global_rank %d, "
                "model comm group %d, group_rank %d, seed comm group id %d, using indices[0:10]: %s"
            ),
            os.getpid(),
            self.dataset.label,
            self.dataset.worker_id,
            self.dataset.global_rank,
            self.dataset.model_comm_group_id,
            self.dataset.model_comm_group_rank,
            self.dataset.sample_comm_group_id,
            samples[:10],
        )
        return samples
