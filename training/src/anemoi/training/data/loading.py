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
from collections.abc import Iterator

import numpy as np
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torchdata.stateful_dataloader.sampler import BatchSampler
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from anemoi.models.distributed.shapes import ShardSizes
from anemoi.training.data.data_reader import BaseAnemoiReader
from anemoi.training.data.multidataset import MultiDataset
from anemoi.training.utils.time_indices import TimeIndices
from anemoi.training.utils.variables_metadata import check_datasets_variables_compatibility


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

    def check_dataset_compatibility(self, dataset: Dataset, **options: object) -> None:
        """Check compatibility requirements specific to this loading policy."""
        del dataset, options


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


class CrossDataset(ConcatDataset):
    """Concatenated domains exposing the dataset metadata used by Anemoi."""

    def __init__(self, datasets: list[MultiDataset]) -> None:
        super().__init__(datasets)
        self.data_readers = {name: reader for dataset in datasets for name, reader in dataset.data_readers.items()}

    @property
    def statistics(self) -> dict:
        return {name: reader.statistics for name, reader in self.data_readers.items()}

    @property
    def metadata(self) -> dict:
        return {name: reader.metadata for name, reader in self.data_readers.items()}

    @property
    def supporting_arrays(self) -> dict:
        return {name: reader.supporting_arrays for name, reader in self.data_readers.items()}

    @property
    def data(self) -> dict:
        return {name: reader.data for name, reader in self.data_readers.items()}

    @property
    def name_to_index(self) -> dict:
        return {name: reader.name_to_index for name, reader in self.data_readers.items()}

    def set_relative_date_indices(self, relative_date_indices: dict[str, TimeIndices]) -> None:
        for dataset in self.datasets:
            dataset.set_relative_date_indices(
                {name: relative_date_indices[name] for name in dataset.data_readers},
            )
        self.cumulative_sizes = self.cumsum(self.datasets)

    def set_reader_group_info(self, reader_group_rank: int, shard_sizes: dict[str, ShardSizes]) -> None:
        for dataset in self.datasets:
            dataset.set_reader_group_info(reader_group_rank, shard_sizes)


class CrossDatasetLoadingPolicy(DataLoadingPolicy):
    """Each sample comes from exactly one domain; batches are domain-homogeneous."""

    def build_dataset(
        self,
        data_readers: dict[str, BaseAnemoiReader],
        relative_date_indices: dict[str, TimeIndices],
        *,
        fake_dataloading: bool = False,
    ) -> ConcatDataset:
        domains = [
            MultiDataset(
                {name: reader},
                {name: relative_date_indices[name]},
                fake_dataloading=fake_dataloading,
            )
            for name, reader in data_readers.items()
        ]
        return CrossDataset(domains)

    def build_batch_sampler(
        self,
        dataset: ConcatDataset,
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        seed: int,
        batch_size: int,
        drop_last: bool,
    ) -> "CrossDatasetBatchSampler":
        return CrossDatasetBatchSampler(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            batch_size=batch_size,
            drop_last=drop_last,
        )

    def check_dataset_compatibility(self, dataset: CrossDataset, **options: object) -> None:
        check_datasets_variables_compatibility(
            {name: data.typed_variables for name, data in dataset.data.items()},
            **options,
        )


class CrossDatasetBatchSampler(Sampler[list[int]]):
    """Randomly interleave whole, domain-homogeneous batches of independently sampled domains.

    The domain order is a shuffle of every domain's batches that depends only on the seed and the
    epoch, so all ranks visit the same domain at each step.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        seed: int,
        batch_size: int,
        drop_last: bool,
    ) -> None:
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self._epoch = 0
        self._domain_offsets = [0, *dataset.cumulative_sizes[:-1]]
        self._domain_samplers = [
            StatefulDistributedSampler(
                domain,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
                seed=seed,
                drop_last=drop_last,
            )
            for domain in dataset.datasets
        ]
        self._batches_yielded = 0

    def _domain_order(self) -> list[int]:
        """Domain of each batch of the epoch, shuffled reproducibly from the seed and the epoch."""
        order = np.repeat(np.arange(len(self._domain_samplers)), self._domain_batch_counts())
        if self.shuffle:
            np.random.default_rng([self.seed, self._epoch]).shuffle(order)
        return order.tolist()

    def _domain_batch_counts(self) -> list[int]:
        return [-(-len(sampler) // self.batch_size) for sampler in self._domain_samplers]

    def __iter__(self) -> Iterator[list[int]]:
        domain_batches = [
            iter(BatchSampler(sampler, batch_size=self.batch_size, drop_last=False))
            for sampler in self._domain_samplers
        ]
        for domain_index in self._domain_order()[self._batches_yielded :]:
            local_batch = next(domain_batches[domain_index])
            self._batches_yielded += 1
            offset = self._domain_offsets[domain_index]
            yield [offset + index for index in local_batch]
        self._batches_yielded = 0

    def __len__(self) -> int:
        return sum(self._domain_batch_counts())

    def set_epoch(self, epoch: int) -> None:
        for sampler in self._domain_samplers:
            sampler.set_epoch(epoch)
        self._epoch = epoch

    def state_dict(self) -> dict:
        return {
            "domain_samplers": [sampler.state_dict() for sampler in self._domain_samplers],
            "batches_yielded": self._batches_yielded,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        for sampler, sampler_state in zip(
            self._domain_samplers,
            state_dict["domain_samplers"],
            strict=True,
        ):
            sampler.load_state_dict(sampler_state)
        self._batches_yielded = state_dict["batches_yielded"]

    @property
    def sampler(self) -> "CrossDatasetBatchSampler":
        """Expose the epoch setter under the same interface as torchdata's BatchSampler."""
        return self
