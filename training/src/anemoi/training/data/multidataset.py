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
import time
from functools import cached_property

import numpy as np
import torch
from rich.console import Console
from rich.tree import Tree
from torch.utils.data import IterableDataset

from anemoi.models.distributed.balanced_partition import get_balanced_partition_range
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.training.data.data_reader import RelativeTimeReader
from anemoi.training.data.data_reader import create_dataset
from anemoi.training.data.data_reader import dates_to_unix_ns
from anemoi.training.data.relative_time_indices import normalize_explicit_time_indices_config
from anemoi.training.data import usable_indices
from anemoi.training.utils.seeding import get_base_seed
from anemoi.utils.dates import frequency_to_seconds

LOGGER = logging.getLogger(__name__)


class MultiDataset(IterableDataset):
    """Multi-dataset wrapper that returns synchronized samples from multiple data readers."""

    def __init__(
        self,
        data_readers: dict,
        relative_date_indices: list | dict[str, list[int]],
        grid_indices: dict | None = None,
        timestep: str = "6h",
        multistep_window: str | datetime.timedelta | None = None,
        dataset_num_inputs: dict[str, int] | None = None,
        dataset_input_selection: dict[str, str] | None = None,
        explicit_time_indices_by_dataset: dict[str, dict[str, list[int]]] | None = None,
        time_index_mode: str = "dense",
        time_index_anchor_dataset: str | None = None,
        shuffle: bool = True,
        label: str = "multi",
        debug: dict | None = None,
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
        """
        self.data_readers = {
            name: data_reader if hasattr(data_reader, "get_sample") else create_dataset(data_reader)
            for name, data_reader in data_readers.items()
        }
        # Backward-compatible alias used throughout the training stack.
        self.datasets = self.data_readers
        self.grid_indices = grid_indices or {}
        self.label = label
        self.shuffle = shuffle
        self.timestep = timestep
        self.dataset_names = list(self.data_readers.keys())
        self.dataset_num_inputs = {str(name): int(value) for name, value in (dataset_num_inputs or {}).items()}
        self.dataset_input_selection = {
            str(name): str(value).strip().lower() for name, value in (dataset_input_selection or {}).items()
        }
        self.relative_date_indices_are_native = hasattr(relative_date_indices, "items")
        if self.relative_date_indices_are_native:
            self.relative_date_indices_by_dataset = {
                str(name): np.array(sorted({int(idx) for idx in indices}), dtype=np.int64)
                for name, indices in relative_date_indices.items()
            }
            self.model_relative_date_indices = np.array(
                sorted({int(idx) for indices in self.relative_date_indices_by_dataset.values() for idx in indices}),
                dtype=np.int64,
            )
        else:
            self.relative_date_indices_by_dataset = None
            self.model_relative_date_indices = np.array(sorted({int(idx) for idx in relative_date_indices}), dtype=np.int64)
        if len(self.model_relative_date_indices) == 0:
            raise ValueError("`relative_date_indices` cannot be empty.")

        try:
            self.timestep_seconds = frequency_to_seconds(self.timestep)
        except ValueError as e:
            msg = f"Error in timestep, {self.timestep}"
            raise ValueError(msg) from e

        self.multistep_window = multistep_window
        parsed_time_index_mode = str(time_index_mode).strip().lower()
        if parsed_time_index_mode not in {"dense", "sparse", "auto_sparse"}:
            raise ValueError(
                f"`time_index_mode` must be one of ['dense', 'sparse', 'auto_sparse'] "
                f"(got '{parsed_time_index_mode}')."
            )
        self.time_index_mode = parsed_time_index_mode
        self.time_index_anchor_dataset = str(time_index_anchor_dataset) if time_index_anchor_dataset else None
        self.explicit_time_indices_by_dataset = normalize_explicit_time_indices_config(
            explicit_time_indices_by_dataset,
        )
        if self.multistep_window is None:
            self.multistep_window_seconds = None
        elif isinstance(self.multistep_window, datetime.timedelta):
            self.multistep_window_seconds = int(self.multistep_window.total_seconds())
        else:
            self.multistep_window_seconds = frequency_to_seconds(self.multistep_window)
        debug = debug or {}
        self.timing_data_enabled = bool(getattr(debug, "timing_data_enabled", False))
        self.timing_data_every = max(1, int(getattr(debug, "timing_data_every", 50)))
        self.timing_rank0_only = bool(getattr(debug, "timing_rank0_only", True))
        self._timing_sample_counter = 0

        # Build per-dataset model/native relative indices.
        # Model relative indices are in units of `timestep`.
        # Native relative indices are in units of each dataset's native frequency.
        self.input_model_relative_date_indices_by_dataset: dict[str, np.ndarray] = {}
        self.target_model_relative_date_indices_by_dataset: dict[str, np.ndarray] = {}
        self.model_relative_date_indices_by_dataset = self._build_model_relative_indices_by_dataset()
        self.input_data_relative_date_indices_by_dataset = {
            name: (
                model_relative_indices.astype(np.int64, copy=False)
                if self.relative_date_indices_are_native
                else self._to_native_relative_indices(name, model_relative_indices)
            )
            for name, model_relative_indices in self.input_model_relative_date_indices_by_dataset.items()
        }
        self.target_data_relative_date_indices_by_dataset = {
            name: (
                model_relative_indices.astype(np.int64, copy=False)
                if self.relative_date_indices_are_native
                else self._to_native_relative_indices(name, model_relative_indices)
            )
            for name, model_relative_indices in self.target_model_relative_date_indices_by_dataset.items()
        }
        self.data_relative_date_indices_by_dataset = {
            name: (
                model_relative_indices.astype(np.int64, copy=False)
                if self.relative_date_indices_are_native
                else self._to_native_relative_indices(name, model_relative_indices)
            )
            for name, model_relative_indices in self.model_relative_date_indices_by_dataset.items()
        }

        # Backward-compat compatibility helper for places expecting a single array.
        if len(self.dataset_names) > 0:
            self.data_relative_date_indices = self.data_relative_date_indices_by_dataset[self.dataset_names[0]]
        else:
            self.data_relative_date_indices = np.array([], dtype=np.int64)

        self._lazy_init_model_and_reader_group_info()
        self._anchor_dataset_name = self._resolve_time_index_anchor_dataset() if self._resolved_time_index_mode() == "sparse" else None
        self.sample_readers = self._build_sample_readers()
        self.compute_valid_date_indices()
        LOGGER.info(
            "MultiDataset initialized with %d datasets (%s)",
            len(self.datasets),
            ", ".join(self.dataset_names),
        )

    def _lazy_init_model_and_reader_group_info(self) -> None:
        """Lazy initialize model and reader group info."""
        self.model_comm_group_rank = 0
        self.model_comm_num_groups = 1
        self.model_comm_group_id = 0
        self.global_rank = 0

        self.reader_group_rank = 0
        self.reader_group_size = 1
        self.shard_shapes = None

        self.sample_comm_num_groups = 1
        self.sample_comm_group_id = 0

        self.ens_comm_group_rank = 0
        self.ens_comm_num_groups = 1
        self.ens_comm_group_id = 0

        self.n_samples_per_worker = 0
        self.chunk_index_range: np.ndarray | None = None

    def _collect(self, attr_name: str) -> dict:
        """Helper method to collect attributes from all data readers."""
        return {name: getattr(dataset, attr_name) for name, dataset in self.data_readers.items()}

    @cached_property
    def statistics(self) -> dict[str, dict]:
        """Return combined statistics from all datasets."""
        return self._collect("statistics")

    @cached_property
    def statistics_tendencies(self) -> dict[str, dict | None]:
        """Return combined tendency statistics from all datasets."""
        return {name: dataset.statistics_tendencies(self.timestep) for name, dataset in self.datasets.items()}

    @cached_property
    def metadata(self) -> dict[str, dict]:
        """Return combined metadata from all datasets."""
        return self._collect("metadata")

    @cached_property
    def supporting_arrays(self) -> dict[str, dict]:
        """Return combined supporting arrays from all datasets."""
        supporting_arrays = self._collect("supporting_arrays")
        for dataset_name, grid_indices in self.grid_indices.items():
            dataset_arrays = supporting_arrays.get(dataset_name, {})
            supporting_arrays[dataset_name] = dataset_arrays | grid_indices.supporting_arrays
        return supporting_arrays

    @cached_property
    def variables(self) -> dict[str, list[str]]:
        """Return combined variables from all datasets."""
        return self._collect("variables")

    @property
    def data(self) -> dict:
        """Return data from all datasets as dictionary."""
        return self._collect("data")

    @cached_property
    def name_to_index(self) -> dict[str, dict]:
        """Return combined name_to_index mapping from all datasets."""
        return self._collect("name_to_index")

    @cached_property
    def resolution(self) -> dict[str, str]:
        """Return combined resolution from all datasets."""
        return self._collect("resolution")

    @cached_property
    def frequency(self) -> datetime.timedelta:
        """Return reference (highest-cadence) frequency across datasets."""
        freqs = self._collect("frequency")
        if len(freqs) == 0:
            msg = "No datasets available to determine frequency."
            raise ValueError(msg)
        return min(freqs.values(), key=frequency_to_seconds)

    @cached_property
    def frequencies(self) -> dict[str, datetime.timedelta]:
        """Return native dataset frequencies."""
        return self._collect("frequency")

    @cached_property
    def timeincrement(self) -> int:
        """Legacy scalar time increment, valid only for uniform or fully aligned frequencies."""
        try:
            frequency = frequency_to_seconds(self.frequency)
        except ValueError as e:
            msg = f"Error in data frequency, {self.frequency}"
            raise ValueError(msg) from e

        timestep = self.timestep_seconds

        assert timestep % frequency == 0, (
            f"Timestep ({self.timestep} == {timestep}) isn't a "
            f"multiple of data frequency ({self.frequency} == {frequency})."
        )

        LOGGER.info(
            "Timeincrement set to %s for data with frequency, %s, and timestep, %s",
            timestep // frequency,
            frequency,
            timestep,
        )
        return timestep // frequency

    @cached_property
    def valid_date_indices(self) -> np.ndarray:
        return self.compute_valid_date_indices()

    def compute_valid_date_indices(self) -> np.ndarray:
        """Return valid date indices.

        A date t is valid if we can sample the elements t + i
        for every relative_date_index i across all datasets.

        Returns the intersection of valid indices from all datasets.
        """
        valid_date_indices_by_dataset: dict[str, np.ndarray] = {}
        for name, ds in self.datasets.items():
            relative_indices = self.data_relative_date_indices_by_dataset[name]
            valid_date_indices = usable_indices.get_usable_indices(
                ds.missing,
                len(ds.dates),
                relative_indices,
                ds.trajectory_ids if ds.has_trajectories else None,
            )
            valid_date_indices_by_dataset[name] = valid_date_indices

            if len(valid_date_indices) == 0:
                msg = f"No valid date indices found for data reader '{name}': {ds}"
                raise ValueError(msg)

            LOGGER.info("Dataset '%s' has %d valid indices", name, len(valid_date_indices))

        mode = self._resolved_time_index_mode()
        if mode == "dense":
            valid_date_indices_intersection = None
            for valid_date_indices in valid_date_indices_by_dataset.values():
                if valid_date_indices_intersection is None:
                    valid_date_indices_intersection = valid_date_indices
                else:
                    valid_date_indices_intersection = np.intersect1d(
                        valid_date_indices_intersection,
                        valid_date_indices,
                    )

            if len(valid_date_indices_intersection) == 0:
                msg = "No valid date indices found after intersection across all datasets."
                raise ValueError(msg)

            LOGGER.info("MultiDataset has %d valid indices after intersection.", len(valid_date_indices_intersection))
            self._anchor_dataset_name = None
            return valid_date_indices_intersection

        anchor_dataset_name = self._resolve_time_index_anchor_dataset()
        anchor_valid_indices = valid_date_indices_by_dataset[anchor_dataset_name]
        if len(anchor_valid_indices) == 0:
            msg = f"Anchor dataset '{anchor_dataset_name}' has no valid date indices."
            raise ValueError(msg)

        LOGGER.info(
            "MultiDataset using sparse time-index mode '%s' with anchor dataset '%s': %d valid indices.",
            mode,
            anchor_dataset_name,
            len(anchor_valid_indices),
        )
        self._anchor_dataset_name = anchor_dataset_name
        return anchor_valid_indices

    def _resolved_time_index_mode(self) -> str:
        if self.time_index_mode != "auto_sparse":
            return self.time_index_mode

        frequency_seconds = {name: frequency_to_seconds(ds.frequency) for name, ds in self.datasets.items()}
        if len(set(frequency_seconds.values())) == 1:
            return "dense"
        return "sparse"

    def _resolve_time_index_anchor_dataset(self) -> str:
        if self.time_index_anchor_dataset is not None:
            if self.time_index_anchor_dataset not in self.dataset_names:
                raise ValueError(
                    f"`time_index_anchor_dataset` '{self.time_index_anchor_dataset}' is not in datasets: "
                    f"{self.dataset_names}"
                )
            return self.time_index_anchor_dataset

        # Default anchor: highest temporal resolution (smallest frequency interval).
        return min(
            self.dataset_names,
            key=lambda dataset_name: frequency_to_seconds(self.datasets[dataset_name].frequency),
        )

    def set_comm_group_info(
        self,
        global_rank: int,
        model_comm_group_id: int,
        model_comm_group_rank: int,
        model_comm_num_groups: int,
        reader_group_rank: int,
        reader_group_size: int,
        shard_shapes: dict[str, list[int]] | None = None,
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
        shard_shapes : dict[str, list[int]] | None
            Optional shard shapes passed from the distributed strategy.
        """
        self.global_rank = global_rank
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

        self.sample_comm_group_id = model_comm_group_id
        self.sample_comm_num_groups = model_comm_num_groups
        if shard_shapes is not None:
            self.shard_shapes = shard_shapes

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
        mode = self._resolved_time_index_mode()
        anchor_dataset_name = getattr(self, "_anchor_dataset_name", None)
        anchor_dates_ns = dates_to_unix_ns(self.datasets[anchor_dataset_name].dates) if anchor_dataset_name else None

        sample_readers = {}
        for name, dataset in self.datasets.items():
            use_sparse_alignment = mode == "sparse" and anchor_dataset_name is not None and name != anchor_dataset_name
            sample_readers[name] = RelativeTimeReader(
                dataset,
                native_relative_indices=self.data_relative_date_indices_by_dataset[name],
                model_relative_indices=self.model_relative_date_indices_by_dataset[name] if use_sparse_alignment else None,
                timestep_seconds=self.timestep_seconds if use_sparse_alignment else None,
                anchor_dates_ns=anchor_dates_ns if use_sparse_alignment else None,
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
        """Initialize all datasets for this worker."""
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

    def get_sample(self, index: int) -> dict[str, torch.Tensor]:
        self._timing_sample_counter += 1
        log_timing = self.timing_data_enabled and self._timing_sample_counter % self.timing_data_every == 0
        if log_timing and self.timing_rank0_only:
            log_timing = self.global_rank == 0 and self.reader_group_rank == 0
        if log_timing:
            t0 = time.perf_counter()
            per_dataset_ms = {}
            per_dataset_time_indices = {}

        x = {}
        for name, dataset in self.sample_readers.items():
            grid_shard_indices = slice(None)
            if self.shard_shapes is not None and self.shard_shapes.get(name) is not None:
                shard_start, shard_end = get_partition_range(self.shard_shapes[name], self.reader_group_rank)
                grid_shard_indices = slice(shard_start, shard_end)
            time_indices = self._resolve_dataset_time_indices(name, index)
            if log_timing:
                t_ds = time.perf_counter()
            x[name] = dataset.get_sample(index, grid_shard_indices)
            if log_timing:
                per_dataset_ms[name] = (time.perf_counter() - t_ds) * 1e3
                per_dataset_time_indices[name] = self._format_time_indices_for_log(time_indices)

        if log_timing:
            total_ms = (time.perf_counter() - t0) * 1e3
            per_dataset_s = ", ".join(f"{name}={ms:.1f}ms" for name, ms in per_dataset_ms.items())
            per_dataset_t = ", ".join(f"{name}={spec}" for name, spec in per_dataset_time_indices.items())
            LOGGER.info(
                (
                    "Timing data_fetch label=%s rank=%d worker=%s sample=%d total_ms=%.1f "
                    "time_indices=[%s] %s"
                ),
                self.label,
                self.global_rank,
                getattr(self, "worker_id", -1),
                self._timing_sample_counter,
                total_ms,
                per_dataset_t,
                per_dataset_s,
            )

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
        # All datasets will use the same shuffled indices for synchronization
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
        for name, dataset in self.datasets.items():
            subtree = dataset.tree(prefix=name)
            tree.add(subtree)
        return tree

    def _build_model_relative_indices_by_dataset(self) -> dict[str, np.ndarray]:
        """Build model-relative time indices available natively for each dataset."""
        model_relative_seconds = self.model_relative_date_indices * self.timestep_seconds
        dataset_model_indices: dict[str, np.ndarray] = {}
        input_model_indices_by_dataset: dict[str, np.ndarray] = {}
        target_model_indices_by_dataset: dict[str, np.ndarray] = {}

        for name, ds in self.datasets.items():
            dataset_frequency_seconds = frequency_to_seconds(ds.frequency)
            requested_model_indices = (
                self.relative_date_indices_by_dataset.get(name, self.model_relative_date_indices)
                if self.relative_date_indices_by_dataset is not None
                else self.model_relative_date_indices
            )
            explicit_cfg = self.explicit_time_indices_by_dataset.get(name)
            if self.relative_date_indices_are_native:
                model_indices = requested_model_indices.astype(np.int64, copy=False)
                input_model_indices = model_indices
                target_model_indices = np.array([], dtype=np.int64)
            elif explicit_cfg is not None:
                input_model_indices = explicit_cfg["input"].astype(np.int64, copy=False)
                target_model_indices = explicit_cfg["target"].astype(np.int64, copy=False)
                model_indices = np.unique(
                    np.concatenate([input_model_indices, target_model_indices]).astype(np.int64, copy=False),
                ).astype(np.int64, copy=False)
                rel_seconds = model_indices * self.timestep_seconds
                if np.any(rel_seconds % dataset_frequency_seconds != 0):
                    raise ValueError(
                        f"Dataset '{name}' explicit time indices {model_indices.tolist()} are not exact native "
                        f"timestamps for dataset frequency {ds.frequency}."
                    )
            else:
                requested_seconds = requested_model_indices * self.timestep_seconds
                exact_mask = requested_seconds % dataset_frequency_seconds == 0
                model_indices = requested_model_indices[exact_mask]
                model_indices = self._augment_windowed_model_indices(
                    dataset_name=name,
                    model_indices=model_indices.astype(np.int64, copy=False),
                    dataset_frequency_seconds=dataset_frequency_seconds,
                )
                input_model_indices = model_indices.astype(np.int64, copy=False)
                target_model_indices = np.array([], dtype=np.int64)

            if len(model_indices) == 0:
                msg = (
                    f"Dataset '{name}' has no exact native timestamps for requested model-relative "
                    f"indices {requested_model_indices.tolist()} with timestep {self.timestep}."
                )
                raise ValueError(msg)

            dataset_model_indices[name] = model_indices
            input_model_indices_by_dataset[name] = input_model_indices
            target_model_indices_by_dataset[name] = target_model_indices
            LOGGER.info(
                "Dataset '%s' uses %d model-relative indices (%s)",
                name,
                len(model_indices),
                model_indices.tolist(),
            )

        unknown_explicit_dataset_keys = sorted(set(self.explicit_time_indices_by_dataset).difference(set(self.dataset_names)))
        if unknown_explicit_dataset_keys:
            raise ValueError(
                f"`explicit_time_indices_by_dataset` provided for unknown datasets: {unknown_explicit_dataset_keys}. "
                f"Known datasets: {self.dataset_names}"
            )
        if self.relative_date_indices_by_dataset is not None:
            unknown_relative_dataset_keys = sorted(
                set(self.relative_date_indices_by_dataset).difference(set(self.dataset_names)),
            )
            if unknown_relative_dataset_keys:
                raise ValueError(
                    f"`relative_date_indices` provided for unknown datasets: {unknown_relative_dataset_keys}. "
                    f"Known datasets: {self.dataset_names}"
                )

        self.input_model_relative_date_indices_by_dataset = input_model_indices_by_dataset
        self.target_model_relative_date_indices_by_dataset = target_model_indices_by_dataset
        return dataset_model_indices

    def _to_native_relative_indices(self, dataset_name: str, model_relative_indices: np.ndarray) -> np.ndarray:
        dataset_frequency_seconds = frequency_to_seconds(self.datasets[dataset_name].frequency)
        rel_seconds = model_relative_indices * self.timestep_seconds
        if np.any(rel_seconds % dataset_frequency_seconds != 0):
            msg = f"Dataset '{dataset_name}' has non-exact native conversion for indices {model_relative_indices}."
            raise ValueError(msg)
        return (rel_seconds // dataset_frequency_seconds).astype(np.int64, copy=False)

    def _augment_windowed_model_indices(
        self,
        *,
        dataset_name: str,
        model_indices: np.ndarray,
        dataset_frequency_seconds: int,
    ) -> np.ndarray:
        num_inputs = self.dataset_num_inputs.get(dataset_name)
        if num_inputs is None:
            return model_indices
        if self.multistep_window_seconds is None:
            raise ValueError("`dataset_num_inputs` requires `multistep_window` to be set.")
        if self.multistep_window_seconds % dataset_frequency_seconds != 0:
            raise ValueError(
                f"`multistep_window` must be divisible by dataset frequency for '{dataset_name}'.",
            )

        native_window_steps = self.multistep_window_seconds // dataset_frequency_seconds
        if native_window_steps <= 0:
            return model_indices

        selection_mode = self.dataset_input_selection.get(dataset_name, "uniform")
        model_steps_per_native_step, remainder = divmod(dataset_frequency_seconds, self.timestep_seconds)
        if remainder != 0:
            raise ValueError(
                f"Dataset '{dataset_name}' frequency {self.datasets[dataset_name].frequency} is not a multiple "
                f"of timestep {self.timestep}.",
            )

        if selection_mode == "all":
            selected_native_indices = np.arange(0, native_window_steps + 1, dtype=np.int64)
        elif selection_mode == "last":
            selected_native_indices = np.arange(
                max(0, native_window_steps - num_inputs),
                native_window_steps + 1,
                dtype=np.int64,
            )
        elif selection_mode == "uniform":
            if native_window_steps % num_inputs != 0:
                raise ValueError(
                    "`dataset_num_inputs` must divide `multistep_window // dataset_frequency` for uniform selection.",
                )
            stride = native_window_steps // num_inputs
            selected_native_indices = np.arange(0, native_window_steps + 1, stride, dtype=np.int64)
        elif selection_mode == "future":
            start_native_index = int(model_indices.max(initial=0) // model_steps_per_native_step)
            selected_native_indices = np.arange(start_native_index, start_native_index + num_inputs, dtype=np.int64)
        else:
            raise ValueError(
                f"Unsupported dataset_input_selection '{selection_mode}' for dataset '{dataset_name}'.",
            )

        selected_model_indices = (selected_native_indices * model_steps_per_native_step).astype(np.int64, copy=False)
        return np.unique(
            np.concatenate([model_indices.astype(np.int64, copy=False), selected_model_indices]),
        ).astype(np.int64, copy=False)

    def _resolve_dataset_time_indices(self, dataset_name: str, index: int) -> slice | int | list[int]:
        reader = self.sample_readers[dataset_name]
        if reader.uses_sparse_alignment:
            return reader.resolve_sparse_time_indices(index)
        return reader.resolve_dense_time_indices(index)

    @staticmethod
    def _format_time_indices_for_log(time_indices: slice | int | list[int]) -> str:
        if isinstance(time_indices, slice):
            return f"{time_indices.start}:{time_indices.stop}:{time_indices.step}"
        if isinstance(time_indices, int):
            return str(time_indices)
        if len(time_indices) <= 8:
            return str(time_indices)
        return f"{time_indices[:4]}...{time_indices[-2:]}(n={len(time_indices)})"
