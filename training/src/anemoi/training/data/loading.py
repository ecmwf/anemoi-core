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
from torch.utils.data import Sampler
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from anemoi.training.data.data_reader import BaseAnemoiReader
from anemoi.training.data.multidataset import MultiDataset
from anemoi.training.utils.time_indices import TimeIndices


class DataLoadingPolicy(ABC):
    """Builds the dataset and sampler consumed by one dataloader stage.

    A policy owns both the dataset layout (e.g. one synchronized dataset vs.
    several independently indexed ones) and the sampler that orders/shards its
    samples, so the two always stay consistent with each other.
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
        """

    @abstractmethod
    def build_sampler(
        self,
        dataset: Dataset,
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        seed: int,
        drop_last: bool,
    ) -> Sampler:
        """Build the sampler that orders and shards the dataset's samples.

        Parameters
        ----------
        dataset : Dataset
            Dataset built by :meth:`build_dataset` for the same stage.
        num_replicas : int
            Number of sample groups the dataset is split across.
        rank : int
            Sample group this sampler produces indices for.
        shuffle : bool
            Shuffle the samples.
        seed : int
            Seed for the shuffle, shared by every rank of a sample group.
        drop_last : bool
            Drop the remaining samples if they don't divide evenly across sample groups.
        """


class SynchronizedLoadingPolicy(DataLoadingPolicy):
    """Default policy: one synchronized dataset and a stock distributed sampler.

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

    def build_sampler(
        self,
        dataset: Dataset,
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        seed: int,
        drop_last: bool,
    ) -> StatefulDistributedSampler:
        return StatefulDistributedSampler(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
