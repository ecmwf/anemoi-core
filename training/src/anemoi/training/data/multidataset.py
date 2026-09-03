# (C) Copyright 2024-2026 Anemoi contributors.
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

import numpy as np
import torch

from anemoi.models.distributed.balanced_partition import get_balanced_partition_range
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.training.data.anemoidataset import AnemoiDataset
from anemoi.training.data.data_reader import BaseAnemoiReader
from anemoi.training.data.usable_indices import compute_valid_anchors
from anemoi.training.utils.seeding import SeedContext
from anemoi.training.utils.seeding import derive_seed
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
            epoch=epoch,
            rollout=rollout,
        )

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

        # Compute valid (sequence, position) anchors and a flat index over them
        # that the shuffle/shard logic operates on.
        self.anchors = compute_valid_anchors(self.data_readers, relative_date_indices)
        self.valid_date_indices = np.arange(len(self.anchors), dtype=np.int64)

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

        # Recompute valid (sequence, position) anchors for the updated rollout.
        self.anchors = compute_valid_anchors(self.data_readers, relative_date_indices)
        self.valid_date_indices = np.arange(len(self.anchors), dtype=np.int64)

        # Normalize the date indices to use slices where possible.
        self.relative_date_indices = {
            name: normalize_time_indices(indices) for name, indices in relative_date_indices.items()
        }

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

    def get_sample(self, index: int) -> dict[str, torch.Tensor]:
        sequence, position = (int(v) for v in self.anchors[index])
        x = {}
        for name, dataset in self.data_readers.items():
            time_steps = offset_time_indices(position, self.relative_date_indices[name])
            # self.shard_sizes is lazily initalised to None
            # This if statement guards against the case where shard_sizes is not set
            # (e.g. if set_comm_group_info hasn't been called yet)
            if self.shard_sizes is not None and self.shard_sizes[name] is not None:
                start, end = get_partition_range(self.shard_sizes[name], self.reader_group_rank)
                grid_indices = slice(start, end)
            else:
                grid_indices = slice(None)
            x[name] = dataset.get_sample(sequence, time_steps, grid_indices)

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
