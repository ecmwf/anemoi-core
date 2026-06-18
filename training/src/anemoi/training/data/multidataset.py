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

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.training.data.data_reader import BaseAnemoiReader
from anemoi.training.data.sampling import SamplingStrategy
from anemoi.training.data.sampling import SynchronizedSampling
from anemoi.training.utils.seeding import get_base_seed
from anemoi.training.utils.time_indices import TimeIndices

LOGGER = logging.getLogger(__name__)


class MultiDataset(IterableDataset):
    """Multi-dataset wrapper that returns samples from multiple data readers.

    The sampling behaviour is determined by the ``sampling_strategy``:

    * ``SynchronizedSampling`` (default) – all readers are sampled at the same
      date index (intersection of valid indices).
    * ``IndependentSampling`` – each reader has its own index pool and
      samples are interleaved randomly.
    """

    def __init__(
        self,
        data_readers: dict[str, BaseAnemoiReader],
        relative_date_indices: dict[str, TimeIndices],
        sampling_strategy: SamplingStrategy | None = None,
        shuffle: bool = True,
        label: str = "multi",
        epoch: int = 0,
        rollout: int = 1,
    ) -> None:
        """Initialize multi-dataset with data readers and a sampling strategy.

        Parameters
        ----------
        data_readers : dict[str, BaseAnemoiReader]
            Dictionary mapping dataset names to their data_readers
            Format: {"dataset_a": data_reader_a, "dataset_b": data_reader_b, ...}
        relative_date_indices : dict[str, TimeIndices]
            Precomputed relative date indices for each data reader
        sampling_strategy : SamplingStrategy | None, optional
            Strategy controlling how samples are drawn. Defaults to ``SynchronizedSampling``.
        shuffle : bool, optional
            Shuffle batches, by default True
        label : str, optional
            label for the dataset, by default "multi"
        epoch : int, optional
            Epoch used for deterministic epoch-dependent shuffling, by default 0
        rollout : int, optional
            Rollout length represented by the loaded relative date indices, by default 1
        """
        self.data_readers = data_readers
        self.sampling_strategy = sampling_strategy or SynchronizedSampling()
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
        self.valid_date_indices = self.sampling_strategy.compute_valid_indices(
            self.data_readers,
            relative_date_indices,
        )

        # Normalize the date indices to use slices where possible.
        self.relative_date_indices = self.sampling_strategy.normalize_date_indices(relative_date_indices)


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

        self.shard_sizes = None

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

    def set_comm_group_info(
        self,
        global_rank: int,
        model_comm_group_id: int,
        model_comm_group_rank: int,
        model_comm_num_groups: int,
        reader_group_rank: int,
        reader_group_size: int,
        shard_sizes: dict[str, ShardSizes],
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
        shard_sizes : dict[str, ShardSizes]
            Shard sizes for all datasets
        """
        self.global_rank = global_rank
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

        self.sample_comm_group_id = model_comm_group_id
        self.sample_comm_num_groups = model_comm_num_groups

        self.shard_sizes = shard_sizes

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

        self.chunk_index_range, self.n_samples_per_worker = self.sampling_strategy.init_worker_chunks(
            self.valid_date_indices,
            self.sample_comm_num_groups,
            self.sample_comm_group_id,
            n_workers,
            worker_id,
        )

        LOGGER.info(
            "Worker %d (pid %d, global_rank %d, model comm group %d)",
            worker_id,
            os.getpid(),
            self.global_rank,
            self.model_comm_group_id,
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

    def get_sample(self, *args) -> dict[str, torch.Tensor]:
        """Get a sample from the data readers, delegating to the sampling strategy.

        Parameters
        ----------
        *args
            For ``SynchronizedSampling``: a single date index (int).
            For ``IndependentSampling``: domain_name (str) and date index (int).

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping dataset name(s) to tensor samples.
        """
        if len(args) == 1:
            index = args[0]
        else:
            index = tuple(args)
        return self.sampling_strategy.get_sample(
            index,
            self.data_readers,
            self.relative_date_indices,
            self.shard_sizes,
            self.reader_group_rank,
        )

    def __iter__(self) -> dict[str, torch.Tensor]:
        """Return an iterator that yields sample dictionaries.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping dataset names to their tensor samples.
        """
        LOGGER.debug(
            "%s worker pid %d, worker id %d",
            self.__class__.__name__,
            os.getpid(),
            self.worker_id,
        )

        yield from self.sampling_strategy.iterate(
            self.valid_date_indices,
            self.chunk_index_range,
            self.shuffle,
            self.rng,
            self.get_sample,
        )

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

