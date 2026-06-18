# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
import random
from collections.abc import Generator

import numpy as np
import torch

from anemoi.models.distributed.balanced_partition import get_balanced_partition_range
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.training.data.anemoidataset import AnemoiDataset
from anemoi.training.data.data_reader import BaseAnemoiReader
from anemoi.training.data.usable_indices import compute_valid_data_indices
from anemoi.training.utils.seeding import get_base_seed
from anemoi.training.utils.time_indices import TimeIndices
from anemoi.training.utils.time_indices import normalize_time_indices
from anemoi.training.utils.time_indices import offset_time_indices

LOGGER = logging.getLogger(__name__)


class MultiDataset(AnemoiDataset):
    """Multi-dataset wrapper that returns synchronized samples from multiple data readers."""

    def __init__(
        self,
        data_readers: dict[str, BaseAnemoiReader],
        relative_date_indices: dict[str, TimeIndices],
        shuffle: bool = True,
        label: str = "multi",
        epoch: int = 0,
        rollout: int = 1,
    ) -> None:
        """Initialize multi-dataset with synchronized data readers.

        Parameters
        ----------
        data_readers : dict[str, BaseAnemoiReader]
            Dictionary mapping dataset names to their data_readers
            Format: {"dataset_a": data_reader_a, "dataset_b": data_reader_b, ...}
        relative_date_indices : dict[str, TimeIndices]
            Precomputed relative date indices for each data reader
        shuffle : bool, optional
            Shuffle batches, by default True
        label : str, optional
            label for the dataset, by default "multi"
        epoch : int, optional
            Epoch used for deterministic epoch-dependent shuffling, by default 0
        rollout : int, optional
            Rollout length represented by the loaded relative date indices, by default 1
        """
        super().__init__(
            data_readers=data_readers,
            shuffle=shuffle,
            label=label,
        )
        self.data_readers = data_readers
        self.label = label
        self.shuffle = shuffle
        self.dataset_names = list(data_readers.keys())
        self.epoch = epoch
        self.rollout = rollout
        self.set_epoch(epoch, rollout=rollout, relative_date_indices=relative_date_indices)

        self._lazy_init_model_and_reader_group_info()

    def set_epoch(
        self,
        epoch: int,
        *,
        rollout: int | None = None,
        relative_date_indices: dict[str, TimeIndices] | None = None,
    ) -> None:
        """Set epoch-dependent sampling state before DataLoader workers are launched."""
        self.epoch = epoch
        if rollout is not None:
            self.rollout = rollout
        if relative_date_indices is None:
            return

        # Refresh which sample dates can provide the currently required time steps.
        self.valid_date_indices = compute_valid_data_indices(self.data_readers, relative_date_indices)

        # Normalize the date indices to use slices where possible, which can improve downstream indexing performance.
        self.relative_date_indices = {
            name: normalize_time_indices(indices) for name, indices in relative_date_indices.items()
        }

        self._lazy_init_model_and_reader_group_info()

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        """Initialize a specific worker.

        Args:
            n_workers : int
                The total number of workers.
            worker_id : int
                The ID of the current worker (0-indexed).

        Returns
        -------
            None
        """
        self.worker_id = worker_id

        # 1. divide valid date indices into shards for sample communication groups (DDP ranks)
        # note that we need even splits here across DDP ranks, so we might throw away some samples
        shard_size = len(self.valid_date_indices) // self.sample_comm_num_groups
        shard_start = self.sample_comm_group_id * shard_size

        self.n_samples_per_worker = shard_size // n_workers

        # 2. partition the shard across workers (here we can have uneven splits, so we use a balanced partition)
        low, high = get_balanced_partition_range(shard_size, n_workers, worker_id, offset=shard_start)

        self.chunk_index_range = np.arange(low, high, dtype=np.uint32)

        LOGGER.info(
            "Worker %d (pid %d, global_rank %d, model comm group %d)  has low/high range %d / %d",
            worker_id,
            os.getpid(),
            self.global_rank,
            self.model_comm_group_id,
            low,
            high,
        )

        base_seed = get_base_seed()
        seed = base_seed + self.epoch

        torch.manual_seed(seed)
        random.seed(seed)
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        sanity_rnd = self.rng.random(1)[0]
        LOGGER.info(
            ("Worker %d (%s, pid %d, epoch %d, rollout %d, seed %d, sanity rnd %f)"),
            worker_id,
            self.label,
            os.getpid(),
            self.epoch,
            self.rollout,
            seed,
            sanity_rnd,
        )

    def get_sample(self, index: int) -> dict[str, torch.Tensor]:
        """Get a sample from the specified domain and index.

        Args:
            domain_name (str): The name of the domain to sample from.
            index (int): The index of the sample to retrieve.

        Returns
        -------
            torch.Tensor: The sample retrieved from the specified domain and index.
        """
        x = {}
        for name, dataset in self.data_readers.items():
            time_steps = offset_time_indices(index, self.relative_date_indices[name])
            # self.shard_shapes is lazily initalised to None
            # This if statement guards against the case where shard_shapes is not set
            # (e.g. if set_comm_group_info hasn't been called yet)
            if self.shard_shapes is not None and self.shard_shapes[name] is not None:
                start, end = get_partition_range(self.shard_shapes[name], self.reader_group_rank)
                grid_indices = slice(start, end)
            else:
                grid_indices = slice(None)
            x[name] = dataset.get_sample(time_steps, grid_indices)

        return x

    def __iter__(self) -> Generator[dict[str, torch.Tensor], None, None]:
        """Return an iterator that yields dictionaries of synchronized samples.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping dataset names to their tensor samples
            Format: {"dataset_a": tensor_a, "dataset_b": tensor_b, ...}
        """
        # Get the shuffled indices from the primary dataset
        # All data readers will use the same shuffled indices for synchronization
        if self.shuffle:
            shuffled_chunk_indices = self.rng.choice(
                self.valid_date_indices,
                size=len(self.valid_date_indices),
                replace=False,
            )[self.chunk_index_range]
        else:
            shuffled_chunk_indices = self.valid_date_indices[self.chunk_index_range]

        LOGGER.debug(
            "%s worker pid %d, worker id %d, using synchronized indices[0:10]: %s",
            self.__class__.__name__,
            os.getpid(),
            self.worker_id,
            shuffled_chunk_indices[:10],
        )

        # TODO(): improve this...
        for i in shuffled_chunk_indices:
            yield self.get_sample(i)
