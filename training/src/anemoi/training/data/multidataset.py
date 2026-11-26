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
from functools import cached_property

import torch
from rich.console import Console
from rich.tree import Tree
from torch.utils.data import Dataset

from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.training.data.data_reader import BaseAnemoiReader
from anemoi.training.data.usable_indices import compute_valid_anchors
from anemoi.training.utils.time_indices import TimeIndices
from anemoi.training.utils.time_indices import normalize_time_indices
from anemoi.training.utils.time_indices import offset_time_indices

LOGGER = logging.getLogger(__name__)


class MultiDataset(Dataset):
    """Multi-dataset wrapper that returns synchronized samples from multiple data readers.

    Each index refers to one valid (sequence, position) anchor shared by all data
    readers. The dataloader's sampler decides which indices a rank loads and in
    which order.
    """

    def __init__(
        self,
        data_readers: dict[str, BaseAnemoiReader],
        relative_date_indices: dict[str, TimeIndices],
        fake_dataloading: bool = False,
    ) -> None:
        """Initialize multi-dataset with synchronized data readers.

        Parameters
        ----------
        data_readers : dict[str, BaseAnemoiReader]
            Dictionary mapping dataset names to their data_readers
            Format: {"dataset_a": data_reader_a, "dataset_b": data_reader_b, ...}
        relative_date_indices : dict[str, TimeIndices]
            Precomputed relative date indices for each data reader
        fake_dataloading : bool, optional
            Load one real sample and reuse it for subsequent accesses, by default False
        """
        self.data_readers = data_readers
        self.dataset_names = list(data_readers.keys())
        self.fake_dataloading = fake_dataloading
        self._fake_sample: dict[str, torch.Tensor] | None = None
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

        self.set_relative_date_indices(relative_date_indices)

        # Reader group info, set by the DDP strategies. Each rank in a reader group
        # reads its own part of the grid.
        self.reader_group_rank = 0
        self.shard_sizes: dict[str, ShardSizes] | None = None

    def set_relative_date_indices(self, relative_date_indices: dict[str, TimeIndices]) -> None:
        """Set the time steps loaded for each sample, and the samples that provide them.

        Call this before the dataloader is created: the number of samples changes
        with the time steps, and the sampler reads it when it is constructed.
        """
        # Compute valid (sequence, position) anchors; each dataset index refers to one anchor.
        self.anchors = compute_valid_anchors(self.data_readers, relative_date_indices)

        # Normalize the date indices to use slices where possible.
        self.relative_date_indices = {
            name: normalize_time_indices(indices) for name, indices in relative_date_indices.items()
        }

    def __len__(self) -> int:
        """Number of valid samples."""
        return len(self.anchors)

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

    def set_reader_group_info(
        self,
        reader_group_rank: int,
        shard_sizes: dict[str, ShardSizes],
    ) -> None:
        """Set the reader group information, called by the DDP strategies.

        Parameters
        ----------
        reader_group_rank : int
            Reader group rank
        shard_sizes : dict[str, ShardSizes]
            Shard sizes for all datasets
        """
        self.reader_group_rank = reader_group_rank
        self.shard_sizes = shard_sizes

    def get_sample(self, index: int) -> dict[str, torch.Tensor]:
        sequence, position = (int(v) for v in self.anchors[index])
        x = {}
        for name, dataset in self.data_readers.items():
            time_steps = offset_time_indices(position, self.relative_date_indices[name])
            # self.shard_sizes is lazily initalised to None
            # This if statement guards against the case where shard_sizes is not set
            # (e.g. if set_reader_group_info hasn't been called yet)
            if self.shard_sizes is not None and self.shard_sizes[name] is not None:
                start, end = get_partition_range(self.shard_sizes[name], self.reader_group_rank)
                grid_indices = slice(start, end)
            else:
                grid_indices = slice(None)
            x[name] = dataset.get_sample(sequence, time_steps, grid_indices)

        return x

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return the synchronized samples of all data readers for one anchor.

        Parameters
        ----------
        index : int
            Index of the anchor, in the range [0, len(self)).

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping dataset names to their tensor samples
            Format: {"dataset_a": tensor_a, "dataset_b": tensor_b, ...}
        """
        if not self.fake_dataloading:
            return self.get_sample(index)
        # Each worker process reads one real sample and then returns it for every index.
        if self._fake_sample is None:
            self._fake_sample = self.get_sample(index)
        return self._fake_sample

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
