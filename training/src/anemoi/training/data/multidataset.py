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

import numpy as np
import torch
from rich.console import Console
from rich.tree import Tree
from torch.utils.data import Dataset

from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.training.data.data_reader import BaseAnemoiReader
from anemoi.training.data.data_reader import RelativeTimeReader
from anemoi.training.data.data_reader import dates_to_unix_ns
from anemoi.training.data.usable_indices import compute_valid_anchors
from anemoi.training.utils.time_indices import TimeIndices
from anemoi.training.utils.time_indices import normalize_time_indices
from anemoi.training.utils.time_indices import offset_time_indices
from anemoi.utils.dates import frequency_to_seconds

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
        self.relative_date_indices_are_native = True
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


class MixedFrequencyDataset(MultiDataset):
    """Synchronized map-style dataset aligned on the highest-frequency reader."""

    def __init__(
        self,
        data_readers: dict[str, BaseAnemoiReader],
        relative_date_indices: list[int],
        frequency: str | datetime.timedelta,
        fake_dataloading: bool = False,
    ) -> None:
        self.data_readers = data_readers
        self.dataset_names = list(data_readers)
        self.relative_date_indices_are_native = False
        self.frequency_seconds = frequency_to_seconds(frequency)
        self.model_relative_date_indices = np.array([], dtype=np.int64)
        self.model_relative_date_indices_by_dataset: dict[str, np.ndarray] = {}
        self.data_relative_date_indices_by_dataset: dict[str, np.ndarray] = {}
        self.sample_readers: dict[str, RelativeTimeReader] = {}
        self.fake_dataloading = fake_dataloading
        self._fake_sample: dict[str, torch.Tensor] | None = None
        if self.fake_dataloading:
            LOGGER.info("Using fake dataloading")

        multi_sequence = [name for name, reader in data_readers.items() if reader.num_sequences > 1]
        if multi_sequence:
            msg = (
                "Mixed-frequency alignment currently supports only datasets with a global time axis. "
                f"Trajectory datasets are unsupported: {multi_sequence}."
            )
            raise ValueError(msg)

        self.reader_group_rank = 0
        self.shard_sizes: dict[str, ShardSizes] | None = None
        self.set_relative_date_indices(relative_date_indices)

    def set_relative_date_indices(self, relative_date_indices: list[int]) -> None:
        """Set shared model-grid indices and resolve each reader's native subset."""
        self.model_relative_date_indices = np.array(
            sorted({int(index) for index in relative_date_indices}),
            dtype=np.int64,
        )
        if len(self.model_relative_date_indices) == 0:
            msg = "`relative_date_indices` cannot be empty."
            raise ValueError(msg)

        self.model_relative_date_indices_by_dataset = self._build_model_relative_indices_by_dataset()
        self.data_relative_date_indices_by_dataset = {
            name: self._to_native_relative_indices(name, model_relative_indices)
            for name, model_relative_indices in self.model_relative_date_indices_by_dataset.items()
        }
        self.relative_date_indices = {
            name: normalize_time_indices(indices.tolist())
            for name, indices in self.data_relative_date_indices_by_dataset.items()
        }
        self._anchor_dataset_name = min(
            self.dataset_names,
            key=lambda name: frequency_to_seconds(self.data_readers[name].frequency),
        )
        anchor_reader = self.data_readers[self._anchor_dataset_name]
        self.anchors = anchor_reader.compute_anchors(
            self.data_relative_date_indices_by_dataset[self._anchor_dataset_name],
        )
        if len(self.anchors) == 0:
            msg = f"Anchor dataset '{self._anchor_dataset_name}' has no valid anchors."
            raise ValueError(msg)

        anchor_dates_ns = dates_to_unix_ns(anchor_reader.dates)
        if anchor_dates_ns is None:
            msg = f"Anchor dataset '{self._anchor_dataset_name}' does not expose datetime dates."
            raise ValueError(msg)

        valid_anchors = np.ones(len(self.anchors), dtype=bool)
        anchor_positions = self.anchors[:, 1]
        for name, reader in self.data_readers.items():
            if name == self._anchor_dataset_name:
                continue

            reader_dates_ns = dates_to_unix_ns(reader.dates)
            if reader_dates_ns is None:
                msg = f"Dataset '{name}' does not expose datetime dates."
                raise ValueError(msg)
            available_mask = np.ones(len(reader_dates_ns), dtype=bool)
            missing = np.asarray(sorted(reader.missing), dtype=np.int64)
            available_mask[missing[(missing >= 0) & (missing < len(available_mask))]] = False
            available_dates_ns = reader_dates_ns[available_mask]
            if len(available_dates_ns) == 0:
                msg = f"Dataset '{name}' has no available dates."
                raise ValueError(msg)

            requested_dates_ns = anchor_dates_ns[anchor_positions, None] + (
                self.model_relative_date_indices_by_dataset[name][None, :] * self.frequency_seconds * 1_000_000_000
            )
            previous_positions = np.searchsorted(available_dates_ns, requested_dates_ns, side="right") - 1
            clipped_positions = np.clip(previous_positions, 0, len(available_dates_ns) - 1)
            ages_ns = requested_dates_ns - available_dates_ns[clipped_positions]
            tolerance_ns = frequency_to_seconds(reader.frequency) * 1_000_000_000
            valid_anchors &= np.all((previous_positions >= 0) & (ages_ns <= tolerance_ns), axis=1)

        self.anchors = self.anchors[valid_anchors]
        if len(self.anchors) == 0:
            msg = "No valid anchors found after mixed-frequency alignment across all datasets."
            raise ValueError(msg)

        self.sample_readers = {
            name: RelativeTimeReader(
                reader,
                native_relative_indices=self.data_relative_date_indices_by_dataset[name],
                model_relative_indices=(
                    self.model_relative_date_indices_by_dataset[name] if name != self._anchor_dataset_name else None
                ),
                frequency_seconds=self.frequency_seconds if name != self._anchor_dataset_name else None,
                anchor_dates_ns=anchor_dates_ns if name != self._anchor_dataset_name else None,
            )
            for name, reader in self.data_readers.items()
        }
        self._fake_sample = None
        LOGGER.info(
            "MixedFrequencyDataset aligned %d samples on '%s'.",
            len(self.anchors),
            self._anchor_dataset_name,
        )

    @cached_property
    def frequency(self) -> datetime.timedelta:
        """Return the highest native frequency across readers."""
        return min(self._collect("frequency").values(), key=frequency_to_seconds)

    def get_sample(self, index: int) -> dict[str, torch.Tensor]:
        sequence, position = (int(value) for value in self.anchors[index])
        sample = {}
        for name in self.data_readers:
            if self.shard_sizes is not None and self.shard_sizes[name] is not None:
                start, end = get_partition_range(self.shard_sizes[name], self.reader_group_rank)
                grid_indices = slice(start, end)
            else:
                grid_indices = slice(None)
            sample[name] = self.sample_readers[name].get_sample(sequence, position, grid_indices)
        return sample

    def _build_model_relative_indices_by_dataset(self) -> dict[str, np.ndarray]:
        """Build model-relative time indices available natively for each dataset."""
        dataset_model_indices: dict[str, np.ndarray] = {}

        for name, ds in self.data_readers.items():
            dataset_frequency_seconds = frequency_to_seconds(ds.frequency)
            requested_model_indices = self.model_relative_date_indices
            requested_seconds = requested_model_indices * self.frequency_seconds
            exact_mask = requested_seconds % dataset_frequency_seconds == 0
            model_indices = requested_model_indices[exact_mask]

            if len(model_indices) == 0:
                msg = (
                    f"Dataset '{name}' has no exact native timestamps for requested model-relative "
                    f"indices {requested_model_indices.tolist()} with frequency {self.frequency_seconds} seconds."
                )
                raise ValueError(msg)

            dataset_model_indices[name] = model_indices
            LOGGER.info(
                "Dataset '%s' uses %d model-relative indices (%s)",
                name,
                len(model_indices),
                model_indices.tolist(),
            )

        return dataset_model_indices

    def _to_native_relative_indices(self, dataset_name: str, model_relative_indices: np.ndarray) -> np.ndarray:
        dataset_frequency_seconds = frequency_to_seconds(self.data_readers[dataset_name].frequency)
        assert self.frequency_seconds is not None
        rel_seconds = model_relative_indices * self.frequency_seconds
        if np.any(rel_seconds % dataset_frequency_seconds != 0):
            msg = f"Dataset '{dataset_name}' has non-exact native conversion for indices {model_relative_indices}."
            raise ValueError(msg)
        return (rel_seconds // dataset_frequency_seconds).astype(np.int64, copy=False)
