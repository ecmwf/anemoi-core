# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import ABC
from abc import abstractmethod

from torch.utils.data import Dataset
from torchdata.stateful_dataloader.sampler import BatchSampler
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from anemoi.training.data.data_reader import BaseAnemoiReader
from anemoi.training.data.multidataset import MultiDataset
from anemoi.training.utils.time_indices import TimeIndices


class DataLoadingPolicy(ABC):
    """Builds the dataset and batch sampler consumed by one dataloader stage.

    A policy owns both the dataset layout (e.g. one synchronized dataset vs.
    several independently indexed ones) and the batch sampler that orders, shards
    and batches its samples, so the two always stay consistent with each other.
    """

    @abstractmethod
    def build_dataset(
        self,
        data_readers: dict[str, BaseAnemoiReader],
        relative_date_indices: dict[str, TimeIndices],
        *,
        fake_dataloading: bool = False,
    ) -> Dataset:
        """Build the dataset for one dataloader stage.

        Parameters
        ----------
        data_readers : dict[str, BaseAnemoiReader]
            Dictionary mapping dataset names to their data readers.
        relative_date_indices : dict[str, TimeIndices]
            Precomputed relative date indices for each data reader.
        fake_dataloading : bool, optional
            Load one real sample and reuse it for subsequent accesses, by default False.

        Returns
        -------
        Dataset
            Map-style dataset that returns one sample for each index.
        """

    @abstractmethod
    def build_batch_sampler(
        self,
        dataset: Dataset,
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        seed: int,
        batch_size: int,
        drop_last: bool,
    ) -> BatchSampler:
        """Build the batch sampler that orders, shards and batches the dataset's samples.

        The returned batch sampler must meet two conditions:

        * Its ``sampler`` attribute has a ``set_epoch(epoch)`` method that selects
          the shuffle of an epoch. The datamodule calls it when it builds a
          dataloader, and Lightning calls it at the start of every epoch.
        * Its batches depend only on the seed and the epoch. A resumed run restores
          the position saved by torchdata's stateful ``BatchSampler``, or, if the
          batch sampler keeps no state, replays the batches of the epoch and skips
          the ones already trained on. Both give the same remaining batches only if
          the order is reproducible.

        Parameters
        ----------
        dataset : Dataset
            Dataset built by :meth:`build_dataset` for the same stage.
        num_replicas : int
            Number of sample groups the dataset is split across.
        rank : int
            Sample group this batch sampler produces batches for.
        shuffle : bool
            Shuffle the samples.
        seed : int
            Seed for the shuffle, shared by every rank of a sample group.
        batch_size : int
            Number of samples per batch.
        drop_last : bool
            Drop the remaining samples if they don't divide evenly across sample groups.

        Returns
        -------
        BatchSampler
            Batch sampler that hands out lists of sample indices for this sample group.
        """


class SynchronizedLoadingPolicy(DataLoadingPolicy):
    """Default policy: one synchronized dataset and a distributed sampler split into batches.

    Every sample contains data from every reader, and the samplers of a sample
    group all hand out the same indices in the same (shuffled) order.
    """

    def build_dataset(
        self,
        data_readers: dict[str, BaseAnemoiReader],
        relative_date_indices: dict[str, TimeIndices],
        *,
        fake_dataloading: bool = False,
    ) -> MultiDataset:
        return MultiDataset(
            data_readers=data_readers,
            relative_date_indices=relative_date_indices,
            fake_dataloading=fake_dataloading,
        )

    def build_batch_sampler(
        self,
        dataset: Dataset,
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        seed: int,
        batch_size: int,
        drop_last: bool,
    ) -> BatchSampler:
        sampler = StatefulDistributedSampler(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        # drop_last in StatefulDistributedSampler shards the dataset evenly across sample groups;
        # the BatchSampler wrapping it only groups a group's own share into batches, and should
        # always keep its last, possibly incomplete, batch.
        return BatchSampler(sampler, batch_size=batch_size, drop_last=False)
