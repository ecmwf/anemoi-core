# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from abc import ABC
from abc import abstractmethod

import numpy as np

from anemoi.models.distributed.balanced_partition import get_balanced_partition_range
from anemoi.training.data.data_reader import BaseAnemoiReader
from anemoi.training.data.usable_indices import compute_valid_data_indices
from anemoi.training.utils.time_indices import TimeIndices
from anemoi.training.utils.time_indices import normalize_time_indices
from anemoi.training.utils.time_indices import offset_time_indices

LOGGER = logging.getLogger(__name__)


class SamplingStrategy(ABC):
    """Base class for dataset sampling strategies."""

    def __init__(self) -> None:
        self.per_dataset_valid_indices = None
        self.valid_indices = None
        self.relative_date_indices = None

    def setup(self, data_readers: dict[str, BaseAnemoiReader], relative_date_indices: dict[str, TimeIndices]) -> None:
        """Optionally perform setup after valid date indices are computed."""
        # Refresh which sample dates can provide the currently required time steps.
        self.per_dataset_valid_indices = compute_valid_data_indices(data_readers, relative_date_indices)
        self.valid_indices = self.compute_valid_indices(self.per_dataset_valid_indices)

        # Normalize the date indices to use slices where possible.
        self.relative_date_indices = {
            name: normalize_time_indices(indices) for name, indices in relative_date_indices.items()
        }

    def init_worker_chunks(
        self,
        sample_comm_num_groups: int,
        sample_comm_group_id: int,
        n_workers: int,
        worker_id: int,
    ) -> tuple[np.ndarray, int]:
        shard_size = self.num_valid_indices // sample_comm_num_groups
        shard_start = sample_comm_group_id * shard_size
        n_samples_per_worker = shard_size // n_workers

        low, high = get_balanced_partition_range(shard_size, n_workers, worker_id, offset=shard_start)
        chunk_index_range = np.arange(low, high, dtype=np.uint32)
        return chunk_index_range, n_samples_per_worker

    def get_sample_indices(self, rng: np.random.Generator, shuffle: bool = True) -> np.ndarray:
        """Get the sample indices for this worker."""
        # Get the shuffled indices from the primary dataset
        # All data readers will use the same shuffled indices for synchronization
        if shuffle:
            return rng.choice(self.num_valid_indices, size=self.num_valid_indices, replace=False)

        return np.arange(self.num_valid_indices)

    @property
    def num_valid_indices(self) -> int:
        return len(self.valid_indices)

    @abstractmethod
    def compute_valid_indices(self, per_dataset_valid_indices: dict[str, np.ndarray]) -> list:
        pass

    @abstractmethod
    def get_sample_timesteps(self, sample_index: int) -> dict[str, np.ndarray]:
        pass


class SynchronizedSampling(SamplingStrategy):
    """All data readers are sampled at the same date index.

    Valid indices are the intersection across all readers, so every
    yielded sample contains data from all readers at the same point.
    """

    def compute_valid_indices(self, per_dataset_valid_indices: dict[str, np.ndarray]) -> list:
        valid_indices = np.intersect1d(*per_dataset_valid_indices.values(), assume_unique=True)
        if len(valid_indices) == 0:
            msg = f"No common valid date indices found across all data readers: {per_dataset_valid_indices}"
            raise ValueError(msg)
        return valid_indices.tolist()

    def get_sample_timesteps(self, sample_index: int) -> dict[str, np.ndarray]:
        assert self.valid_indices is not None, f"{self.__class__.__name__}.setup() must be called before sampling."
        valid_sample_index = self.valid_indices[sample_index]
        indices = {}
        for dataset_name, relative_indices in self.relative_date_indices.items():
            indices[dataset_name] = offset_time_indices(valid_sample_index, relative_indices)
        return indices


class VariableSampling(SamplingStrategy):
    """Each data reader maintains its own valid-index pool.

    Samples from different readers are interleaved randomly, and each
    yielded dict contains data from exactly 1 reader.

    This strategy is used for multi-domain training, where the model
    is trained with different domains.
    """

    def compute_valid_indices(self, per_dataset_valid_indices: dict[str, np.ndarray]) -> list:
        """Compute valid indices for variable sampling."""
        return [
            (dataset_name, idx) for dataset_name, indices in per_dataset_valid_indices.items() for idx in indices
        ]

    def get_sample_timesteps(self, sample_index: int) -> dict[str, np.ndarray]:
        assert self.valid_indices is not None, f"{self.__class__.__name__}.setup() must be called before sampling."
        dataset_name, valid_sample_index = self.valid_indices[sample_index]
        relative_indices = self.relative_date_indices[dataset_name]
        return {dataset_name: offset_time_indices(valid_sample_index, relative_indices)}
