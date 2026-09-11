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

import numpy as np
import torch

from anemoi.training.data.data_reader import BaseAnemoiReader
from anemoi.training.data.datasets import AnemoiDataset
from anemoi.training.data.usable_indices import compute_valid_anchors
from anemoi.training.utils.time_indices import TimeIndices

LOGGER = logging.getLogger(__name__)


class MultiDataset(AnemoiDataset):
    """Multi-dataset wrapper that returns synchronized samples from multiple data readers.

    Selection policy: ONE set of anchors shared by all readers (intersection of
    the readers' valid anchors); every sample reads ALL readers at that anchor.
    """

    def __init__(
        self,
        data_readers: dict[str, BaseAnemoiReader],
        relative_date_indices: dict[str, TimeIndices],
        shuffle: bool = True,
        label: str = "multi",
        epoch: int = 0,
        rollout: int = 1,
        batch_size: int = 1,
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
        batch_size : int, optional
            Per-GPU batch size, by default 1. Not used by this dataset (every sample
            already contains all readers); accepted for a uniform constructor.
        """
        super().__init__(
            data_readers=data_readers,
            shuffle=shuffle,
            label=label,
            epoch=epoch,
            rollout=rollout,
            batch_size=batch_size,
        )
        self._check_no_mixed_sequence_types()
        self._set_relative_date_indices(relative_date_indices)

    def _compute_anchors(self, relative_date_indices: dict[str, TimeIndices]) -> None:
        # Valid (sequence, position) anchors shared by all readers, plus a flat index
        # over them that the shuffle/shard logic operates on.
        self.anchors = compute_valid_anchors(self.data_readers, relative_date_indices)
        self.valid_date_indices = np.arange(len(self.anchors), dtype=np.int64)

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        """Initialize all data readers for this worker."""
        self.worker_id = worker_id
        # divide valid date indices into shards for sample communication groups (DDP ranks)
        # note that we need even splits here across DDP ranks, so we might throw away some samples
        self.n_samples_per_worker, self.chunk_index_range = self._shard_indices(
            len(self.valid_date_indices),
            n_workers,
            worker_id,
        )
        self._seed_worker()

    def get_sample(self, index: int) -> dict[str, torch.Tensor]:
        sequence, position = (int(v) for v in self.anchors[index])
        return {name: self._read(name, sequence, position) for name in self.data_readers}

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
