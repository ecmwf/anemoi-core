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
import os
import random
from collections.abc import Mapping
from functools import cached_property

import numpy as np
import torch
from hydra.utils import instantiate
from rich.console import Console
from rich.tree import Tree
from torch.utils.data import IterableDataset

from anemoi.models.distributed.balanced_partition import get_balanced_partition_range
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.training.data.data_reader import BaseAnemoiReader
from anemoi.training.data.iteration import BaseIteration
from anemoi.training.utils.seeding import SeedContext
from anemoi.training.utils.seeding import derive_seed
from anemoi.training.utils.seeding import get_base_seed
from anemoi.training.utils.time_indices import TimeIndices
from anemoi.training.utils.time_indices import normalize_time_indices

LOGGER = logging.getLogger(__name__)


class MultiDataset(IterableDataset):
    """Multi-dataset wrapper that returns synchronized samples from multiple data readers."""

    def __init__(
        self,
        data_readers: dict[str, BaseAnemoiReader],
        relative_date_indices: dict[str, TimeIndices],
        shuffle: bool = True,
        label: str = "multi",
        epoch: int = 0,
        rollout: int = 1,
        fake_dataloading: bool = False,
        iteration: Mapping[str, object] | None = None,
        check_dataset_units: bool = False,
        check_variables_compatibility: Mapping[str, object] | None = None,
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
        fake_dataloading : bool, optional
            Load one real sample and reuse it for subsequent accesses, by default False
        iteration : Mapping[str, object], optional
            Hydra configuration for dataset iteration, by default ``BaseIteration``
        check_dataset_units : bool, optional
            Check common variable metadata for compatibility, by default False
        check_variables_compatibility : Mapping[str, object], optional
            Options forwarded to ``Variable.check_compatibility`` when
            ``check_dataset_units`` is enabled.
        """
        self.data_readers = data_readers
        self.label = label
        self.shuffle = shuffle
        self.dataset_names = list(data_readers.keys())
        self.epoch = epoch
        self.rollout = rollout
        self.fake_dataloading = fake_dataloading
        if self.fake_dataloading:
            LOGGER.info("Using fake dataloading")

        # Guard against mixing single-sequence (NativeGridDataset, global time axis)
        # with multi-sequence (TrajectoryDataset, init x step axes).  The anchor
        # intersection would silently keep only sequence-0 samples and produce
        # semantically meaningless alignment between the two encoders.
        single_seq = [n for n, ds in data_readers.items() if ds.num_sequences == 1]
        multi_seq = [n for n, ds in data_readers.items() if ds.num_sequences > 1]
        if single_seq and multi_seq:
            msg = (
                "Currently mixing single-sequence datasets (global time axis) with "
                "Trajectory datasets (init x step axes) in the same MultiDataset is unsupported. "
                f"Single-sequence: {single_seq}. Trajectory: {multi_seq}. "
            )
            raise ValueError(msg)

        self.iteration = BaseIteration() if iteration is None else instantiate(iteration)
        self._set_date_indices(relative_date_indices)

        self._lazy_init_model_and_reader_group_info()
        if check_dataset_units:
            self._check_datasets_units(**dict(check_variables_compatibility or {}))

    def _set_date_indices(self, relative_date_indices: dict[str, TimeIndices]) -> None:
        """Set anchors and relative date indices."""
        initializing = not hasattr(self, "valid_date_indices")
        self.anchors, self.valid_date_indices = self.iteration.compute_anchors(self, relative_date_indices)
        if initializing and isinstance(self.valid_date_indices, Mapping):
            LOGGER.info("valid date indices: %s", self.valid_date_indices)

        # Normalize the date indices to use slices where possible.
        self.relative_date_indices = {
            name: normalize_time_indices(indices) for name, indices in relative_date_indices.items()
        }

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

        self._set_date_indices(relative_date_indices)

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

    def _set_chunk_and_workers(self, n_workers: int, worker_id: int) -> None:
        """Set the sample count and chunk indices assigned to a worker."""
        index_groups = (
            self.valid_date_indices.items()
            if isinstance(self.valid_date_indices, Mapping)
            else ((None, self.valid_date_indices),)
        )
        samples_per_worker = {}
        chunk_index_range = {}
        for name, indices in index_groups:
            # 1. divide valid date indices into shards for sample communication groups (DDP ranks)
            # note that we need even splits here across DDP ranks, so we might throw away some samples
            shard_size = len(indices) // self.sample_comm_num_groups
            shard_start = self.sample_comm_group_id * shard_size
            samples_per_worker[name] = shard_size // n_workers

            # 2. partition the shard across workers (here we can have uneven splits, so we use a balanced partition)
            low, high = get_balanced_partition_range(shard_size, n_workers, worker_id, offset=shard_start)
            chunk_index_range[name] = np.arange(low, high, dtype=np.uint32)
            LOGGER.info(
                "Worker %d (pid %d, global_rank %d, model comm group %d)  has low/high range %d / %d",
                worker_id,
                os.getpid(),
                self.global_rank,
                self.model_comm_group_id,
                low,
                high,
            )

        if None in samples_per_worker:
            self.n_samples_per_worker = samples_per_worker[None]
            self.chunk_index_range = chunk_index_range[None]
        else:
            self.n_samples_per_worker = samples_per_worker
            self.chunk_index_range = chunk_index_range

    def _check_datasets_units(self, **options: object) -> None:
        """Check common variables for compatibility across datasets."""
        from anemoi.transform.variables import Variable

        dataset_variables = {}
        for name, data in self.data.items():
            if variables := data.typed_variables:
                dataset_variables[name] = variables
        if len(dataset_variables) == 0:
            LOGGER.warning("All datasets have empty metadata, skipping units check.")
            return
        if len(dataset_variables) == 1:
            LOGGER.warning("Only one dataset has variable metadata, skipping units check.")
            return

        dataset_names = list(dataset_variables)
        for index, dataset_name in enumerate(dataset_names):
            for other_name in dataset_names[index + 1 :]:
                common_variables = dataset_variables[dataset_name].keys() & dataset_variables[other_name].keys()
                try:
                    Variable.check_compatibility(
                        {name: dataset_variables[dataset_name][name] for name in common_variables},
                        {name: dataset_variables[other_name][name] for name in common_variables},
                        **options,
                    )
                except ValueError as error:
                    msg = (
                        f"Variable compatibility check failed for domain1 '{dataset_name}' "
                        f"and domain2 '{other_name}': {error}"
                    )
                    raise ValueError(msg) from error

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        """Initialize all data readers for this worker."""
        self.worker_id = worker_id
        self._set_chunk_and_workers(n_workers, worker_id)

        base_seed = get_base_seed()
        # The datamodule checkpoints this epoch and restores it before new workers
        # start, so resuming from an epoch checkpoint derives the same seed.
        seed = derive_seed(base_seed, SeedContext.DATALOADER, self.epoch)

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

    def __iter__(self) -> dict[str, torch.Tensor]:
        """Return an iterator that yields dictionaries of synchronized samples.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping dataset names to their tensor samples
            Format: {"dataset_a": tensor_a, "dataset_b": tensor_b, ...}
        """
        yield from self.iteration(self)

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
