# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Sampling strategies for multi-dataset training.

Two strategies are provided:

* ``SynchronizedSampling`` – every data reader is sampled at the **same**
  date index in each iteration step.  Valid indices are the *intersection*
  across all readers.
* ``IndependentSampling`` – each data reader maintains its **own** pool of
  valid indices.  Samples from different readers are interleaved randomly.
"""

import logging
from abc import ABC
from abc import abstractmethod
from collections.abc import Generator

import numpy as np
import torch

from anemoi.models.distributed.balanced_partition import get_balanced_partition_range
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.training.data.data_reader import BaseAnemoiReader
from anemoi.training.data.usable_indices import compute_valid_data_indices
from anemoi.training.data.usable_indices import get_usable_indices
from anemoi.training.utils.time_indices import TimeIndices
from anemoi.training.utils.time_indices import normalize_time_indices
from anemoi.training.utils.time_indices import offset_time_indices

LOGGER = logging.getLogger(__name__)


class SamplingStrategy(ABC):
    """Base class for dataset sampling strategies."""

    @abstractmethod
    def compute_valid_indices(
        self,
        data_readers: dict[str, BaseAnemoiReader],
        relative_date_indices: dict[str, TimeIndices],
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Compute valid date indices for the given data readers."""

    @abstractmethod
    def init_worker_chunks(
        self,
        valid_date_indices: np.ndarray | dict[str, np.ndarray],
        sample_comm_num_groups: int,
        sample_comm_group_id: int,
        n_workers: int,
        worker_id: int,
    ) -> tuple:
        """Partition valid indices across DDP ranks and dataloader workers.

        Returns
        -------
        tuple
            (chunk_index_range, n_samples_per_worker)
        """

    @abstractmethod
    def iterate(
        self,
        valid_date_indices: np.ndarray | dict[str, np.ndarray],
        chunk_index_range: np.ndarray | dict[str, np.ndarray],
        shuffle: bool,
        rng: np.random.Generator,
        get_sample_fn: callable,
    ) -> Generator[dict[str, torch.Tensor], None, None]:
        """Yield samples according to the strategy."""

    @abstractmethod
    def get_sample(
        self,
        index: int | tuple[str, int],
        data_readers: dict[str, BaseAnemoiReader],
        relative_date_indices: dict[str, TimeIndices],
        shard_shapes: dict[str, list[int]] | None,
        reader_group_rank: int,
    ) -> dict[str, torch.Tensor]:
        """Retrieve one sample from the data readers."""

    @staticmethod
    def normalize_date_indices(
        relative_date_indices: dict[str, TimeIndices],
    ) -> dict[str, TimeIndices]:
        """Normalize relative date indices to use slices where possible."""
        return {name: normalize_time_indices(indices) for name, indices in relative_date_indices.items()}


class SynchronizedSampling(SamplingStrategy):
    """All data readers are sampled at the same date index.

    Valid indices are the intersection across all readers, so every
    yielded sample contains data from *all* readers at the same point.
    """

    def compute_valid_indices(
        self,
        data_readers: dict[str, BaseAnemoiReader],
        relative_date_indices: dict[str, TimeIndices],
    ) -> np.ndarray:
        return compute_valid_data_indices(data_readers, relative_date_indices)

    def init_worker_chunks(
        self,
        valid_date_indices: np.ndarray,
        sample_comm_num_groups: int,
        sample_comm_group_id: int,
        n_workers: int,
        worker_id: int,
    ) -> tuple[np.ndarray, int]:
        shard_size = len(valid_date_indices) // sample_comm_num_groups
        shard_start = sample_comm_group_id * shard_size
        n_samples_per_worker = shard_size // n_workers

        low, high = get_balanced_partition_range(shard_size, n_workers, worker_id, offset=shard_start)
        chunk_index_range = np.arange(low, high, dtype=np.uint32)
        return chunk_index_range, n_samples_per_worker

    def iterate(
        self,
        valid_date_indices: np.ndarray,
        chunk_index_range: np.ndarray,
        shuffle: bool,
        rng: np.random.Generator,
        get_sample_fn: callable,
    ) -> Generator[dict[str, torch.Tensor], None, None]:
        if shuffle:
            shuffled_chunk_indices = rng.choice(
                valid_date_indices,
                size=len(valid_date_indices),
                replace=False,
            )[chunk_index_range]
        else:
            shuffled_chunk_indices = valid_date_indices[chunk_index_range]

        for i in shuffled_chunk_indices:
            yield get_sample_fn(i)

    def get_sample(
        self,
        index: int,
        data_readers: dict[str, BaseAnemoiReader],
        relative_date_indices: dict[str, TimeIndices],
        shard_shapes: dict[str, list[int]] | None,
        reader_group_rank: int,
    ) -> dict[str, torch.Tensor]:
        x = {}
        for name, dataset in data_readers.items():
            time_steps = offset_time_indices(index, relative_date_indices[name])
            if shard_shapes is not None and shard_shapes[name] is not None:
                start, end = get_partition_range(shard_shapes[name], reader_group_rank)
                grid_indices = slice(start, end)
            else:
                grid_indices = slice(None)
            x[name] = dataset.get_sample(time_steps, grid_indices)
        return x


class IndependentSampling(SamplingStrategy):
    """Each data reader maintains its own valid-index pool.

    Samples from different readers are interleaved randomly, and each
    yielded dict contains data from exactly **one** reader.
    """

    def compute_valid_indices(
        self,
        data_readers: dict[str, BaseAnemoiReader],
        relative_date_indices: dict[str, TimeIndices],
    ) -> dict[str, np.ndarray]:
        return {
            name: get_usable_indices(
                ds.missing,
                len(ds.dates),
                relative_date_indices[name],
                ds.trajectory_ids if ds.has_trajectories else None,
            )
            for name, ds in data_readers.items()
        }

    def init_worker_chunks(
        self,
        valid_date_indices: dict[str, np.ndarray],
        sample_comm_num_groups: int,
        sample_comm_group_id: int,
        n_workers: int,
        worker_id: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        chunk_index_range = {}
        n_samples_per_worker = {}
        for dataset, indices in valid_date_indices.items():
            shard_size = len(indices) // sample_comm_num_groups
            shard_start = sample_comm_group_id * shard_size
            n_samples_per_worker[dataset] = shard_size // n_workers

            low, high = get_balanced_partition_range(shard_size, n_workers, worker_id, offset=shard_start)
            chunk_index_range[dataset] = np.arange(low, high, dtype=np.uint32)
        return chunk_index_range, n_samples_per_worker

    def iterate(
        self,
        valid_date_indices: dict[str, np.ndarray],
        chunk_index_range: dict[str, np.ndarray],
        shuffle: bool,
        rng: np.random.Generator,
        get_sample_fn: callable,
    ) -> Generator[dict[str, torch.Tensor], None, None]:
        if shuffle:
            shuffled_chunk_indices = {
                dataset: rng.choice(indices, size=len(indices), replace=False)[chunk_index_range[dataset]]
                for dataset, indices in valid_date_indices.items()
            }
            labeled_samples_and_indexes = [
                (domain, i) for domain, indices in shuffled_chunk_indices.items() for i in indices
            ]
            labeled_samples = rng.choice(
                labeled_samples_and_indexes,
                size=len(labeled_samples_and_indexes),
                replace=False,
            )
        else:
            shuffled_chunk_indices = {
                domain: indices[chunk_index_range[domain]] for domain, indices in valid_date_indices.items()
            }
            labeled_samples = [(domain, i) for domain, inds in shuffled_chunk_indices.items() for i in inds]

        for domain_name, index in labeled_samples:
            yield get_sample_fn(domain_name, int(index))

    def get_sample(
        self,
        index: tuple[str, int],
        data_readers: dict[str, BaseAnemoiReader],
        relative_date_indices: dict[str, TimeIndices],
        shard_shapes: dict[str, list[int]] | None,
        reader_group_rank: int,
    ) -> dict[str, torch.Tensor]:
        domain_name, date_index = index
        time_step = offset_time_indices(date_index, relative_date_indices[domain_name])
        if shard_shapes is not None and shard_shapes[domain_name] is not None:
            start, end = get_partition_range(shard_shapes[domain_name], reader_group_rank)
            grid_indices = slice(start, end)
        else:
            grid_indices = slice(None)
        return {domain_name: data_readers[domain_name].get_sample(time_step, grid_indices)}
