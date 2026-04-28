# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
import os
import random
from functools import cached_property

import numpy as np
import torch
from rich.console import Console
from rich.tree import Tree
from torch.utils.data import IterableDataset

from anemoi.training.data.data_reader import BaseAnemoiReader
from anemoi.training.utils.time_indices import TimeIndices
from anemoi.training.data.usable_indices import get_usable_indices
from anemoi.training.utils.seeding import get_base_seed
from anemoi.training.data.multidataset import MultiDataset
from anemoi.models.distributed.balanced_partition import (
    get_balanced_partition_range,
    get_partition_range,
)
from anemoi.training.utils.time_indices import offset_time_indices


LOGGER = logging.getLogger(__name__)

# it may be that multidomain is not a good name for this, but it is what it is for now


class MultiDomainDataset(MultiDataset):
    def __init__(
        self,
        data_readers: dict[str, BaseAnemoiReader],
        relative_date_indices: dict[str, TimeIndices],
        shuffle: bool = True,
        label: str = "multidomain",
    ) -> None:
        """
        A dataset that combines multiple datasets together,
        where each dataset is associated with a different "domain".
        The domains are defined by the keys of the `data_readers` and
        `relative_date_indices` dictionaries.

        Args:
            data_readers (dict[str, BaseAnemoiReader]):
                A dictionary mapping domain names to their corresponding data readers.
            relative_date_indices (dict[str, TimeIndices]):
                A dictionary mapping domain names to their corresponding relative date indices.
            shuffle (bool, optional):
                Whether to shuffle the data. Defaults to True.
            label (str, optional):
                A label for this dataset. Defaults to "multidomain".
        Return:
            None
        """
        super().__init__(
            data_readers=data_readers,
            relative_date_indices=relative_date_indices,
            shuffle=shuffle,
            label=label,
        )
        self.n_samples_per_worker = {}  # overwrite base to empty dict
        self.chunk_index_range = {}  # overwrite base to empty dict

    def valid_date_indices(self) -> np.ndarray:
        """
        Does not create an intersection of valid date indices across datasets,
        as the datasets may have different valid date indices. Instead, we compute the valid date indices
        for each dataset separately. This allows us to sample from each dataset independently,
        which is necessary for the multi-domain setting.

        Args:
            None

        Returns:
            dict[str, np.ndarray]: A dictionary mapping domain names to their corresponding valid date indices.
        """
        return {
            name: get_usable_indices(
                ds.missing,
                len(ds.dates),
                self.relative_date_indices[name],
                ds.trajectory_ids if ds.has_trajectories else None,
            )
            for name, ds in self.data_readers.items()
        }

    def per_worker_init(self, n_workers, worker_id):
        self.worker_id = worker_id

        for dataset, indices in self.relative_date_indices.items():
            shard_size = len(self.valid_date_indices) // self.sample_comm_num_groups
            shard_start = self.sample_comm_group_id * shard_size

            self.n_samples_per_worker[dataset] = shard_size // n_workers

            low, high = get_balanced_partition_range(
                shard_size, n_workers, worker_id, offset=shard_start
            )

            self.chunk_index_range[dataset] = np.arange(low, high, dtype=np.uint32)

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
            torch.manual_seed(base_seed)
            random.seed(base_seed)
            self.rng = np.random.default_rng(seed=base_seed)
            sanity_rnd = self.rng.random(1)[0]
            LOGGER.info(
                ("Worker %d (%s, pid %d, base_seed %d, sanity rnd %f)"),
                worker_id,
                self.label,
                os.getpid(),
                base_seed,
                sanity_rnd,
            )

    def get_sample(self, domain_name: str, index: int) -> torch.Tensor:
        """
        Get a sample from the specified domain and index.
        Args:
            domain_name (str): The name of the domain to sample from.
            index (int): The index of the sample to retrieve.
        Returns:
            torch.Tensor: The sample retrieved from the specified domain and index.
        """
        time_step = offset_time_indices(index, self.relative_date_indices[domain_name])
        if self.shard_shapes is not None and self.shard_shapes[domain_name] is not None:
            start, end = get_partition_range(
                self.shard_shapes[domain_name], self.reader_group_rank
            )
            grid_indices = slice(start, end)
        else:
            grid_indices = slice(None)

        return self.data_readers[domain_name].get_sample(time_step, grid_indices)

    def __iter__(self) -> tuple[torch.Tensor, str]:
        if self.shuffle:
            shuffled_chunk_indices = {
                dataset: self.rng.choice(
                    indices,
                    size=len(indices),
                    replace=False,
                )
                for dataset, indices in self.valid_date_indices.items()
            }

            labeled_samples_and_indexes = [
                (domain, i)
                for domain, indices in shuffled_chunk_indices.items()
                for i in indices
            ]

            labeled_samples = self.rng.choice(
                labeled_samples_and_indexes,
                size=len(labeled_samples_and_indexes),
                replace=False,
            )
        else:
            shuffled_chunk_indices = {
                domain: indices[self.chunk_index_range[domain]]
                for domain, indices in self.valid_date_indices.items()
            }
            labeled_samples = [
                (domain, i)
                for domain, inds in shuffled_chunk_indices.items()
                for i in inds
            ]

        LOGGER.debug(
            (
                "Worker pid %d, label %s, worker id %d, global_rank %d, "
                "model comm group %d, group_rank %d, seed comm group id %d, using indices[0:10]: %s"
            ),
            os.getpid(),
            self.label,
            self.worker_id,
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
            self.sample_comm_group_id,
            labeled_samples[:10],
        )

        for batch in labeled_samples:
            domain_name, index = batch
            yield domain_name, self.get_sample(domain_name, index)
