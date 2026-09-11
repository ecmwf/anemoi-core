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
from abc import ABC
from abc import abstractmethod
from collections.abc import Mapping
from functools import cached_property

import numpy as np
import torch
from rich.console import Console
from rich.tree import Tree
from torch.utils.data import IterableDataset

from anemoi.models.distributed.balanced_partition import get_balanced_partition_range
from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.training.data.data_reader import BaseAnemoiReader
from anemoi.training.utils.seeding import SeedContext
from anemoi.training.utils.seeding import derive_seed
from anemoi.training.utils.seeding import get_base_seed
from anemoi.training.utils.time_indices import TimeIndices
from anemoi.training.utils.time_indices import normalize_time_indices
from anemoi.training.utils.time_indices import offset_time_indices

LOGGER = logging.getLogger(__name__)

ParticipantReaders = dict[str, dict[str, BaseAnemoiReader]]
"""Nested readers: ``{dataset_name: {participant_name: reader}}``."""


def normalize_participant_readers(
    data_readers: Mapping[str, BaseAnemoiReader | Mapping[str, BaseAnemoiReader]],
) -> ParticipantReaders:
    """Normalise readers to the nested ``{dataset_name: {participant: reader}}`` form.

    A dataset given as a bare reader (the single form) has exactly one
    participant, named after the dataset. Nested entries are copied as is.
    """
    nested: ParticipantReaders = {}
    for dataset_name, entry in data_readers.items():
        if isinstance(entry, Mapping):
            if not entry:
                msg = f"Dataset '{dataset_name}' has no participants."
                raise ValueError(msg)
            nested[dataset_name] = dict(entry)
        else:
            nested[dataset_name] = {dataset_name: entry}
    return nested


class AnemoiDataset(IterableDataset, ABC):
    """Base Anemoi Datasets torch dataset class.

    Subclasses own the *selection policy* (which anchors exist, how they are
    shuffled and interleaved across readers) by implementing
    :meth:`_compute_anchors`, :meth:`per_worker_init` and :meth:`__iter__`.
    Everything that does not depend on that policy (worker seeding, worker
    sharding, sharded reads, relative-date-index normalisation) lives here.

    Readers are organised as ``{dataset_name: {participant: reader}}``
    (:attr:`participant_readers`). A dataset given as a bare reader has one
    participant named after the dataset. A *participant row*
    (:meth:`participant_row`) is the ``{dataset_name: reader}`` view of one
    participant across all datasets; selection policies work on rows.
    Everything keyed by ``dataset_name`` only (relative date indices, shard
    sizes, statistics, metadata, ...) is a per-dataset quantity taken from the
    dataset's reference participant (:attr:`reference_readers`).
    """

    def __init__(
        self,
        data_readers: Mapping[str, BaseAnemoiReader | Mapping[str, BaseAnemoiReader]],
        shuffle: bool = True,
        label: str = "multi",
        epoch: int = 0,
        rollout: int = 1,
        batch_size: int = 1,
    ) -> None:
        """Initialize a dataset backed by one or more data readers.

        Parameters
        ----------
        data_readers : Mapping[str, BaseAnemoiReader | Mapping[str, BaseAnemoiReader]]
            Dataset names mapped to a reader (single form) or to ``{participant: reader}``.
            Format: {"dataset_a": reader_a, "dataset_b": {"p1": reader_b1, "p2": reader_b2}, ...}
        shuffle : bool, optional
            Shuffle batches, by default True
        label : str, optional
            label for the dataset, by default "multi"
        epoch : int, optional
            Epoch used for deterministic epoch-dependent shuffling, by default 0
        rollout : int, optional
            Rollout length represented by the loaded relative date indices, by default 1
        batch_size : int, optional
            Per-GPU batch size the DataLoader will collate from this dataset, by default 1.
            Subclasses whose sampling order must respect batch boundaries use it.
        """
        if batch_size < 1:
            msg = f"batch_size must be >= 1, got {batch_size}"
            raise ValueError(msg)
        self.participant_readers = normalize_participant_readers(data_readers)
        self.dataset_names = list(self.participant_readers.keys())
        self.label = label
        self.shuffle = shuffle
        self.epoch = epoch
        self.rollout = rollout
        self.batch_size = batch_size
        self.relative_date_indices: dict[str, TimeIndices] = {}
        self._lazy_init_model_and_reader_group_info()

    def participant_row(self, participant: str | None = None) -> dict[str, BaseAnemoiReader]:
        """Return the readers of one participant across all datasets: ``{dataset_name: reader}``.

        A dataset with a single participant takes part in every row. With
        ``participant=None`` every dataset must have exactly one participant
        (there is only one row, the :class:`MultiDataset` case).
        """
        row = {}
        for dataset_name, participants in self.participant_readers.items():
            if participant is not None and participant in participants:
                row[dataset_name] = participants[participant]
            elif len(participants) == 1:
                row[dataset_name] = next(iter(participants.values()))
            else:
                msg = (
                    f"Dataset '{dataset_name}' has participants {list(participants)}, none of which is "
                    f"'{participant}'."
                )
                raise ValueError(msg)
        return row

    @property
    def reference_readers(self) -> dict[str, BaseAnemoiReader]:
        """Return one representative reader per dataset (its first participant).

        All participants of a dataset share frequency and variables, so this is
        the reader to use for per-dataset quantities such as relative date indices,
        statistics and variable indices.
        """
        return {name: next(iter(participants.values())) for name, participants in self.participant_readers.items()}

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
        self.worker_id = 0
        self.seed: int | None = None
        self.rng: np.random.Generator | None = None
        self.n_samples_per_worker = 0
        self.chunk_index_range: np.ndarray | None = None

    def _check_no_mixed_sequence_types(self) -> None:
        """Reject mixing single-sequence readers with multi-sequence (trajectory) readers.

        Anchors of a single-sequence reader (global time axis) and of a
        trajectory reader (init x step axes) are not comparable; combining them
        would silently keep only sequence-0 samples.
        """
        single_seq, multi_seq = [], []
        for dataset_name, participants in self.participant_readers.items():
            for participant, reader in participants.items():
                name = dataset_name if participant == dataset_name else f"{dataset_name}/{participant}"
                (single_seq if reader.num_sequences == 1 else multi_seq).append(name)
        if single_seq and multi_seq:
            msg = (
                "Currently mixing single-sequence datasets (global time axis) with "
                f"Trajectory datasets (init x step axes) in the same {self.__class__.__name__} is unsupported. "
                f"Single-sequence: {single_seq}. Trajectory: {multi_seq}. "
            )
            raise ValueError(msg)

    def _set_relative_date_indices(self, relative_date_indices: dict[str, TimeIndices]) -> None:
        """Recompute anchors and store normalized relative date indices.

        Normalizing to slices where possible improves downstream indexing performance.
        """
        self._compute_anchors(relative_date_indices)
        self.relative_date_indices = {
            name: normalize_time_indices(indices) for name, indices in relative_date_indices.items()
        }

    @abstractmethod
    def _compute_anchors(self, relative_date_indices: dict[str, TimeIndices]) -> None:
        """Compute the valid ``(sequence, position)`` anchors and the flat index over them.

        This is the selection policy of the subclass (e.g. intersection across
        readers vs. independent anchors per reader).
        """

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
        self._set_relative_date_indices(relative_date_indices)

    def _collect(self, attr_name: str) -> dict:
        """Collect ``attr_name`` per dataset, from the dataset's reference reader."""
        return {name: getattr(reader, attr_name) for name, reader in self.reference_readers.items()}

    def _collect_participants(self, attr_name: str) -> dict[str, dict]:
        """Collect ``attr_name`` per dataset and participant: ``{dataset_name: {participant: value}}``."""
        return {
            name: {participant: getattr(reader, attr_name) for participant, reader in participants.items()}
            for name, participants in self.participant_readers.items()
        }

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

    @abstractmethod
    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        """Initialize worker state (shards, seed) for this worker. To be overwritten by subclasses.

        Implementations typically set ``self.worker_id``, call
        :meth:`_shard_indices` for each index set they iterate over and finish
        with :meth:`_seed_worker`.
        """

    def _get_worker_index_range(self, n_samples: int, n_workers: int, worker_id: int) -> tuple[int, int, int]:
        """Partition samples across communication groups and workers."""
        shard_size = n_samples // self.sample_comm_num_groups
        shard_start = self.sample_comm_group_id * shard_size
        low, high = get_balanced_partition_range(shard_size, n_workers, worker_id, offset=shard_start)
        return shard_size // n_workers, low, high

    def _shard_indices(self, n_samples: int, n_workers: int, worker_id: int) -> tuple[int, np.ndarray]:
        """Return this worker's share of ``n_samples`` flat indices.

        Returns
        -------
        tuple[int, np.ndarray]
            Number of samples per worker and the contiguous index range
            ``[low, high)`` assigned to this worker (after sharding across
            sample communication groups).
        """
        n_samples_per_worker, low, high = self._get_worker_index_range(n_samples, n_workers, worker_id)
        LOGGER.info(
            "Worker %d (pid %d, global_rank %d, model comm group %d)  has low/high range %d / %d",
            worker_id,
            os.getpid(),
            self.global_rank,
            self.model_comm_group_id,
            low,
            high,
        )
        return n_samples_per_worker, np.arange(low, high, dtype=np.uint32)

    def _seed_worker(self) -> None:
        """Seed torch, random and this dataset's numpy generator for the current epoch.

        The seed depends on the base seed and the epoch only (no rank or worker
        term), so all ranks and workers draw the same shuffle for the same
        epoch. The datamodule checkpoints the epoch and restores it before new
        workers start, so resuming from a checkpoint derives the same seed.
        """
        base_seed = get_base_seed()
        seed = derive_seed(base_seed, SeedContext.DATALOADER, self.epoch)

        torch.manual_seed(seed)
        random.seed(seed)
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        sanity_rnd = self.rng.random(1)[0]
        LOGGER.info(
            ("Worker %d (%s, pid %d, epoch %d, rollout %d, seed %d, sanity rnd %f)"),
            self.worker_id,
            self.label,
            os.getpid(),
            self.epoch,
            self.rollout,
            seed,
            sanity_rnd,
        )

    def _read(
        self,
        dataset_name: str,
        sequence: int,
        position: int,
        participant: str | None = None,
    ) -> torch.Tensor:
        """Read one sample of ``dataset_name`` at a ``(sequence, position)`` anchor.

        ``participant`` selects the reader within the dataset (default: its
        first, for single-participant datasets its only, participant). Applies
        the dataset's relative date indices and, when shard sizes are known
        (set by ``set_comm_group_info``), restricts the grid to this reader
        group rank's shard.
        """
        participants = self.participant_readers[dataset_name]
        reader = participants[participant] if participant is not None else next(iter(participants.values()))
        time_steps = offset_time_indices(position, self.relative_date_indices[dataset_name])
        # self.shard_sizes is lazily initialised to None; guard against the case where
        # set_comm_group_info has not been called yet.
        if self.shard_sizes is not None and self.shard_sizes[dataset_name] is not None:
            start, end = get_partition_range(self.shard_sizes[dataset_name], self.reader_group_rank)
            grid_indices = slice(start, end)
        else:
            grid_indices = slice(None)
        return reader.get_sample(sequence, time_steps, grid_indices)

    @cached_property
    def shard_shapes(self) -> dict[str, list]:
        """Return shard shapes for all data readers."""
        shard_shapes = {}
        for name, reader in self.reference_readers.items():
            shard_shapes[name] = get_balanced_partition_sizes(reader.grid_size, self.reader_group_size)
        return shard_shapes

    def get_shard_slice(self, dataset_name: str, reader_group_rank: int) -> slice:
        """Get the grid shard slice according to the reader rank."""
        start, end = get_partition_range(
            partition_sizes=self.shard_shapes[dataset_name],
            partition_id=reader_group_rank,
        )
        return slice(start, end)

    @abstractmethod
    def __iter__(self) -> None:
        """Return an iterator over the dataset(s). To be overwritten by subclasses.

        This is the only sampling contract of the base class: how samples are
        addressed (``get_sample`` signature) is left to the subclass.
        """

    def __repr__(self) -> str:
        console = Console(record=True, width=120)
        with console.capture() as capture:
            console.print(self.tree())
        return capture.get()

    def tree(self) -> Tree:
        tree = Tree(f"{self.__class__.__name__}")
        for dataset_name, participants in self.participant_readers.items():
            for participant, reader in participants.items():
                prefix = dataset_name if participant == dataset_name else f"{dataset_name}/{participant}"
                tree.add(reader.tree(prefix=prefix))
        return tree
