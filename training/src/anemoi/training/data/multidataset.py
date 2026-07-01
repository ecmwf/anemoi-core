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

from anemoi.models.distributed.balanced_partition import get_balanced_partition_range
from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.training.data.data_reader import BaseAnemoiReader
from anemoi.training.data.data_reader import RelativeTimeReader
from anemoi.training.data.data_reader import dates_to_unix_ns
from anemoi.training.data.usable_indices import compute_valid_data_indices
from anemoi.training.utils.seeding import get_base_seed
from anemoi.training.utils.time_indices import TimeIndices
from anemoi.training.utils.time_indices import normalize_time_indices
from anemoi.training.utils.time_indices import offset_time_indices
from anemoi.utils.dates import frequency_to_seconds

LOGGER = logging.getLogger(__name__)


class MultiDataset(IterableDataset):
    """Multi-dataset wrapper that returns synchronized samples from multiple data readers."""

    def __init__(
        self,
        data_readers: dict[str, BaseAnemoiReader],
        relative_date_indices: list[int] | dict[str, TimeIndices],
        frequency: str | None = None,
        shuffle: bool = True,
        label: str = "multi",
    ) -> None:
        """Initialize multi-dataset with synchronized data readers.

        Parameters
        ----------
        data_readers : dict[str, BaseAnemoiReader]
            Dictionary mapping dataset names to their data_readers
            Format: {"dataset_a": data_reader_a, "dataset_b": data_reader_b, ...}
        relative_date_indices : list[int] | dict[str, TimeIndices]
            Precomputed relative date indices for each data reader.
            In the mixed-frequency path this is the shared model-relative index window.
        frequency : str | None, optional
            Shared model frequency used to convert model-relative indices to
            each dataset's native time grid. Only needed in the mixed-frequency path.
        shuffle : bool, optional
            Shuffle batches, by default True
        label : str, optional
            label for the dataset, by default "multi"
        """
        self.data_readers = data_readers
        self.label = label
        self.shuffle = shuffle
        self.dataset_names = list(self.data_readers.keys())
        self.relative_date_indices_are_native = hasattr(relative_date_indices, "items")
        self.frequency_seconds: int | None = None
        self.model_relative_date_indices = np.array([], dtype=np.int64)
        self.model_relative_date_indices_by_dataset: dict[str, np.ndarray] = {}
        self.data_relative_date_indices_by_dataset: dict[str, np.ndarray] = {}
        self.raw_relative_date_indices: dict[str, TimeIndices] = {}
        self._anchor_dataset_name: str | None = None
        self.sample_readers: dict[str, RelativeTimeReader] = {}

        if self.relative_date_indices_are_native:
            self.raw_relative_date_indices = dict(relative_date_indices)
            self.relative_date_indices = {
                name: normalize_time_indices(indices) for name, indices in relative_date_indices.items()
            }
            self._lazy_init_model_and_reader_group_info()
            _ = self.valid_date_indices
            return

        if frequency is None:
            msg = "`frequency` is required when `relative_date_indices` are provided on the shared model grid."
            raise ValueError(msg)

        self.model_relative_date_indices = np.array(
            sorted({int(idx) for idx in relative_date_indices}),
            dtype=np.int64,
        )
        if len(self.model_relative_date_indices) == 0:
            msg = "`relative_date_indices` cannot be empty."
            raise ValueError(msg)

        try:
            self.frequency_seconds = frequency_to_seconds(frequency)
        except ValueError as e:
            msg = f"Error in frequency, {frequency}"
            raise ValueError(msg) from e

        # Build per-dataset model/native relative indices.
        # Model relative indices are in units of the shared model frequency.
        # Native relative indices are in units of each dataset's native frequency.
        self.model_relative_date_indices_by_dataset = self._build_model_relative_indices_by_dataset()
        self.data_relative_date_indices_by_dataset = {
            name: self._to_native_relative_indices(name, model_relative_indices)
            for name, model_relative_indices in self.model_relative_date_indices_by_dataset.items()
        }
        self.relative_date_indices = {
            name: normalize_time_indices(indices.tolist())
            for name, indices in self.data_relative_date_indices_by_dataset.items()
        }

        self._lazy_init_model_and_reader_group_info()
        self._anchor_dataset_name = self._resolve_anchor_dataset_name()
        self.sample_readers = self._build_sample_readers()
        _ = self.valid_date_indices
        LOGGER.info(
            "MultiDataset initialized with %d datasets (%s)",
            len(self.data_readers),
            ", ".join(self.dataset_names),
        )

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
        if self.relative_date_indices_are_native:
            freq_ref = None
            for name, freq in freqs.items():
                if freq_ref is None:
                    freq_ref = freq
                assert freq == freq_ref, f"Data reader '{name}' has different frequency than other data readers"
            return freq_ref
        return min(freqs.values(), key=frequency_to_seconds)

    @cached_property
    def valid_date_indices(self) -> np.ndarray:
        return self.compute_valid_date_indices()

    def compute_valid_date_indices(self) -> np.ndarray:
        """Return valid date indices.

        In the native-index path, returns the intersection of valid indices across
        all data readers. In the mixed-frequency path, returns only the anchor
        dataset's (highest-resolution) valid indices; non-anchor datasets provide
        their last available timestep at or before each anchor time.
        """
        if self.relative_date_indices_are_native:
            return compute_valid_data_indices(self.data_readers, self.raw_relative_date_indices)

        valid_date_indices_by_dataset: dict[str, np.ndarray] = {}
        for name, ds in self.data_readers.items():
            relative_indices = self.data_relative_date_indices_by_dataset[name]
            dataset_valid_date_indices = compute_valid_data_indices(
                {name: ds},
                {name: relative_indices},
            )
            valid_date_indices_by_dataset[name] = dataset_valid_date_indices

            if len(dataset_valid_date_indices) == 0:
                msg = f"No valid date indices found for data reader '{name}': {ds}"
                raise ValueError(msg)

        anchor_dataset_name = self._resolve_anchor_dataset_name()
        anchor_valid_indices = valid_date_indices_by_dataset[anchor_dataset_name]
        if len(anchor_valid_indices) == 0:
            msg = f"Anchor dataset '{anchor_dataset_name}' has no valid date indices."
            raise ValueError(msg)

        LOGGER.info(
            "MultiDataset using mixed-frequency time-index alignment with anchor dataset '%s': %d valid indices.",
            anchor_dataset_name,
            len(anchor_valid_indices),
        )
        self._anchor_dataset_name = anchor_dataset_name
        return anchor_valid_indices

    def _resolve_anchor_dataset_name(self) -> str:
        # Default anchor: highest temporal resolution (smallest frequency interval).
        return min(
            self.dataset_names,
            key=lambda dataset_name: frequency_to_seconds(self.data_readers[dataset_name].frequency),
        )

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
            Shard sizes for all data readers
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

    def _build_sample_readers(self) -> dict[str, RelativeTimeReader]:
        anchor_dataset_name = getattr(self, "_anchor_dataset_name", None)
        anchor_dates_ns = (
            dates_to_unix_ns(self.data_readers[anchor_dataset_name].dates) if anchor_dataset_name else None
        )

        sample_readers = {}
        for name, dataset in self.data_readers.items():
            use_mixed_frequency_alignment = anchor_dataset_name is not None and name != anchor_dataset_name
            sample_readers[name] = RelativeTimeReader(
                dataset,
                native_relative_indices=self.data_relative_date_indices_by_dataset[name],
                model_relative_indices=(
                    self.model_relative_date_indices_by_dataset[name] if use_mixed_frequency_alignment else None
                ),
                frequency_seconds=self.frequency_seconds if use_mixed_frequency_alignment else None,
                anchor_dates_ns=anchor_dates_ns if use_mixed_frequency_alignment else None,
            )
        return sample_readers

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

        torch.manual_seed(base_seed)
        random.seed(base_seed)
        self.rng = np.random.default_rng(seed=base_seed)
        sanity_rnd = self.rng.random(1)[0]
        LOGGER.info(
            ("Worker %d (%s, pid %d, base_seed %d, sanity rnd %f)"),
            worker_id,
            self.label,
            os.getpid(),
            base_seed,
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

    def get_sample(self, index: int) -> dict[str, torch.Tensor]:
        x = {}
        for name, dataset in self.data_readers.items():
            if self.shard_sizes is not None and self.shard_sizes[name] is not None:
                grid_indices = self.get_shard_slice(name, self.reader_group_rank)
            else:
                grid_indices = slice(None)
            if self.relative_date_indices_are_native:
                time_steps = offset_time_indices(index, self.relative_date_indices[name])
                x[name] = dataset.get_sample(time_steps, grid_indices)
            else:
                x[name] = self.sample_readers[name].get_sample(index, grid_indices)

        return x

    def __iter__(self) -> dict[str, torch.Tensor]:
        """Return an iterator that yields dictionaries of synchronized samples.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping dataset names to their tensor samples
            Format: {"dataset_a": tensor_a, "dataset_b": tensor_b, ...}
        """
        # Single-process/no-worker dataloaders may skip worker_init_fn.
        if not hasattr(self, "rng") or self.chunk_index_range is None:
            self.per_worker_init(n_workers=1, worker_id=0)

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
