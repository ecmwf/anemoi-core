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

from anemoi.models.distributed.balanced_partition import get_balanced_partition_range
from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.training.data.data_reader import BaseAnemoiReader
from anemoi.training.data.relative_time_indices import offset_time_indices
from anemoi.training.data.usable_indices import get_usable_indices
from anemoi.training.utils.seeding import get_base_seed

LOGGER = logging.getLogger(__name__)


class MultiDataset(IterableDataset):
    """Multi-dataset wrapper that returns synchronized samples from multiple data readers."""

    def __init__(
        self,
        data_readers: dict[str, BaseAnemoiReader],
        relative_date_indices: dict[str, list[int]],
        shuffle: bool = True,
        label: str = "multi",
    ) -> None:
        """Initialize multi-dataset with synchronized data readers.

        Parameters
        ----------
        data_readers : dict[str, BaseAnemoiReader]
            Dictionary mapping dataset names to their data_readers
            Format: {"dataset_a": data_reader_a, "dataset_b": data_reader_b, ...}
        relative_date_indices : dict[str, list[int]]
            Precomputed relative date indices for each data reader
        shuffle : bool, optional
            Shuffle batches, by default True
        label : str, optional
            label for the dataset, by default "multi"
        """
        self.data_readers = data_readers
        self.relative_date_indices = relative_date_indices
        self.label = label
        self.shuffle = shuffle
        self.dataset_names = list(data_readers.keys())

        self._lazy_init_model_and_reader_group_info()

    def _lazy_init_model_and_reader_group_info(self) -> None:
        """Lazy initialize model and reader group info."""
        # lazy init model and reader group info, will be set by the DDPGroupStrategy:
        self.model_comm_group_rank = 0
        self.model_comm_num_groups = 1
        self.model_comm_group_id = 0
        self.global_rank = 0

        self.reader_group_rank = 0
        self.reader_group_size = 1

        self.sample_comm_num_groups = 1  # groups that work on the same sample / batch
        self.sample_comm_group_id = 0

        self.ens_comm_group_rank = 0
        self.ens_comm_num_groups = 1
        self.ens_comm_group_id = 0

        self.shard_shapes = None

        # additional state vars (lazy init)
        self.n_samples_per_worker = 0
        self.chunk_index_range: np.ndarray | None = None

    def _collect(self, attr_name: str) -> dict:
        """Helper method to collect attributes from all data readers."""
        return {name: getattr(dataset, attr_name) for name, dataset in self.data_readers.items()}

    @cached_property
    def statistics(self) -> dict[str, dict]:
        """Return combined statistics from all data readers."""
        return self._collect("statistics")

    @cached_property
    def metadata(self) -> dict[str, dict]:
        """Return combined metadata from all data readers."""
        return self._collect("metadata")

    @cached_property
    def supporting_arrays(self) -> dict[str, dict]:
        """Return combined supporting arrays from all data readers."""
        return self._collect("supporting_arrays")

    @cached_property
    def variables(self) -> dict[str, list[str]]:
        """Return combined variables from all data readers."""
        return self._collect("variables")

    @property
    def data(self) -> dict:
        """Return data from all data readers as dictionary."""
        return self._collect("data")

    @cached_property
    def name_to_index(self) -> dict[str, dict]:
        """Return combined name_to_index mapping from all data readers."""
        return self._collect("name_to_index")

    @cached_property
    def resolution(self) -> dict[str, str]:
        """Return combined resolution from all data readers."""
        return self._collect("resolution")

    @cached_property
    def frequency(self) -> datetime.timedelta:
        """Return combined frequency from all data readers."""
        freqs = self._collect("frequency")
        freq_ref = None
        for name, freq in freqs.items():
            if freq_ref is None:
                freq_ref = freq
            assert freq == freq_ref, f"Data reader '{name}' has different frequency than other data readers"
        return freq_ref

    @cached_property
    def valid_date_indices(self) -> np.ndarray:
        """Return valid date indices.

        A date t is valid if we can sample the elements t + i
        for every relative_date_index i across all data readers.

        Returns the intersection of valid indices from all data readers.
        """
        valid_date_indices_intersection = None
        for dataset_name, ds in self.data_readers.items():
            valid_date_indices = get_usable_indices(
                ds.missing,
                len(ds.dates),
                self.relative_date_indices[dataset_name],
                ds.trajectory_ids if ds.has_trajectories else None,
            )
            if valid_date_indices_intersection is None:
                valid_date_indices_intersection = valid_date_indices
            else:
                valid_date_indices_intersection = np.intersect1d(valid_date_indices_intersection, valid_date_indices)

            if len(valid_date_indices) == 0:
                msg = f"No valid date indices found for data reader '{dataset_name}': {ds}"
                raise ValueError(msg)

            LOGGER.info("Data reader '%s' has %d valid indices", dataset_name, len(valid_date_indices))

        if len(valid_date_indices_intersection) == 0:
            msg = "No valid date indices found after intersection across all datasets."
            raise ValueError(msg)

        LOGGER.info("MultiDataset has %d valid indices after intersection.", len(valid_date_indices_intersection))

        return valid_date_indices_intersection

    def set_comm_group_info(
        self,
        global_rank: int,
        model_comm_group_id: int,
        model_comm_group_rank: int,
        model_comm_num_groups: int,
        reader_group_rank: int,
        reader_group_size: int,
        shard_shapes: dict[str, list[int]],
    ) -> None:
        """Set model and reader communication group information (called by DDPGroupStrategy).

        Parameters
        ----------
        global_rank : int
            Global rank
        model_comm_group_id : int
            Model communication group ID
        model_comm_group_rank : int
            Model communication group rank
        model_comm_num_groups : int
            Number of model communication groups
        reader_group_rank : int
            Reader group rank
        reader_group_size : int
            Reader group size
        shard_shapes : dict[str, list[int]]
            Shard shapes for all data readers
        """
        self.global_rank = global_rank
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

        self.sample_comm_group_id = model_comm_group_id
        self.sample_comm_num_groups = model_comm_num_groups

        self.shard_shapes = shard_shapes

        assert self.reader_group_size >= 1, f"reader_group_size(={self.reader_group_size}) must be positive"

        LOGGER.info(
            "NativeGridDataset.set_group_info(): global_rank %d, model_comm_group_id %d, "
            "model_comm_group_rank %d, model_comm_num_groups %d, reader_group_rank %d, "
            "sample_comm_group_id %d, sample_comm_num_groups %d",
            global_rank,
            model_comm_group_id,
            model_comm_group_rank,
            model_comm_num_groups,
            reader_group_rank,
            self.sample_comm_group_id,
            self.sample_comm_num_groups,
        )

    def set_ens_comm_group_info(
        self,
        ens_comm_group_id: int,
        ens_comm_group_rank: int,
        ens_comm_num_groups: int,
    ) -> None:
        """Set ensemble communication group information (called by DDPGroupStrategy).

        Parameters
        ----------
        ens_comm_group_id : int
            Ensemble communication group ID
        ens_comm_group_rank : int
            Ensemble communication group rank
        ens_comm_num_groups : int
            Number of ensemble communication groups
        """
        self.ens_comm_group_id = ens_comm_group_id
        self.ens_comm_group_rank = ens_comm_group_rank
        self.ens_comm_num_groups = ens_comm_num_groups

        self.sample_comm_group_id = ens_comm_group_id
        self.sample_comm_num_groups = ens_comm_num_groups

        LOGGER.info(
            "NativeGridDataset.set_ens_comm_group_info(): global_rank %d, ens_comm_group_id %d, "
            "ens_comm_group_rank %d, ens_comm_num_groups %d, reader_group_rank %d, "
            "sample_comm_group_id %d, sample_comm_num_groups %d",
            self.global_rank,
            ens_comm_group_id,
            ens_comm_group_rank,
            ens_comm_num_groups,
            self.reader_group_rank,
            self.sample_comm_group_id,
            self.sample_comm_num_groups,
        )

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        """Initialize all data readers for this worker."""
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

    @cached_property
    def shard_shapes(self) -> dict[str, list]:
        """Return shard shapes for all data readers."""
        shard_shapes = {}
        for name, dataset in self.data_readers.items():
            shard_shapes[name] = get_balanced_partition_sizes(dataset.grid_size, self.reader_group_size)
        return shard_shapes

    def get_shard_slice(self, dataset_name: str, reader_group_rank: int) -> slice:
        """Get the grid shard slice according to the reader rank."""
        start, end = get_partition_range(
            partition_sizes=self.shard_shapes[dataset_name],
            partition_id=reader_group_rank,
        )
        return slice(start, end)

    def get_sample(self, index: int) -> dict[str, torch.Tensor]:
        x = {}
        for name, dataset in self.data_readers.items():
            time_steps = offset_time_indices(index, self.relative_time_indexers[name])
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

    def __iter__(self) -> dict[str, torch.Tensor]:
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

    def __repr__(self) -> str:
        console = Console(record=True, width=120)
        with console.capture() as capture:
            console.print(self.tree())
        return capture.get()

    def tree(self) -> Tree:
        tree = Tree(f"{self.__class__.__name__}")
        for name, dataset in self.data_readers.items():
            subtree = dataset.tree(prefix=name)
            tree.add(subtree)
        return tree
