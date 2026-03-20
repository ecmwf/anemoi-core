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
from anemoi.training.data.dataset import create_dataset
from anemoi.training.data.usable_indices import get_usable_indices
from anemoi.training.utils.seeding import get_base_seed
from anemoi.utils.dates import frequency_to_seconds

LOGGER = logging.getLogger(__name__)


class MultiDataset(IterableDataset):
    """Multi-dataset wrapper that returns synchronized samples from multiple data readers."""

    def __init__(
        self,
        data_readers: dict,
        grid_indices: dict,
        relative_date_indices: list,
        timestep: str = "6h",
        multistep_window: str | datetime.timedelta | None = None,
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
        self.data_readers = data_readers
        self.label = label
        self.shuffle = shuffle
        self.timestep = timestep
        self.dataset_names = list(data_readers.keys())
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
        self.explicit_time_indices_by_dataset = self._normalize_explicit_time_indices_config(
            explicit_time_indices_by_dataset,
        )
        debug = debug or {}
        self.timing_data_enabled = bool(getattr(debug, "timing_data_enabled", False))
        self.timing_data_every = max(1, int(getattr(debug, "timing_data_every", 50)))
        self.timing_rank0_only = bool(getattr(debug, "timing_rank0_only", True))
        self._timing_sample_counter = 0

        # Create each dataset
        self.datasets = {name: create_dataset(data_reader) for name, data_reader in data_readers.items()}
        self._dates_ns_by_dataset, self._date_to_native_index_by_dataset = self._build_dataset_date_index_maps()

        # Build per-dataset model/native relative indices.
        # Model relative indices are in units of `timestep`.
        # Native relative indices are in units of each dataset's native frequency.
        self.input_model_relative_date_indices_by_dataset: dict[str, np.ndarray] = {}
        self.target_model_relative_date_indices_by_dataset: dict[str, np.ndarray] = {}
        self.model_relative_date_indices_by_dataset = self._build_model_relative_indices_by_dataset()
        self.input_data_relative_date_indices_by_dataset = {
            name: self._to_native_relative_indices(name, model_relative_indices)
            for name, model_relative_indices in self.input_model_relative_date_indices_by_dataset.items()
        }
        self.target_data_relative_date_indices_by_dataset = {
            name: self._to_native_relative_indices(name, model_relative_indices)
            for name, model_relative_indices in self.target_model_relative_date_indices_by_dataset.items()
        }
        self.data_relative_date_indices_by_dataset = {
            name: self._to_native_relative_indices(name, model_relative_indices)
            for name, model_relative_indices in self.model_relative_date_indices_by_dataset.items()
        }

        # Backward-compat compatibility helper for places expecting a single array.
        if len(self.dataset_names) > 0:
            self.data_relative_date_indices = self.data_relative_date_indices_by_dataset[self.dataset_names[0]]
        else:
            self.data_relative_date_indices = np.array([], dtype=np.int64)

        LOGGER.info(
            "MultiDataset initialized with %d datasets (%s), %d valid indices each",
            len(self.datasets),
            ", ".join(self.dataset_names),
            len(self.valid_date_indices),
        )
        self._lazy_init_model_and_reader_group_info()
        self._sparse_missing_log_counts: dict[str, int] = {}

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

        # additional state vars (lazy init)
        self.n_samples_per_worker = 0
        self.chunk_index_range: np.ndarray | None = None

    def _collect(self, attr_name: str) -> dict:
        """Helper method to collect attributes from all data readers."""
        return {name: getattr(dataset, attr_name) for name, dataset in self.data_readers.items()}

    @staticmethod
    def _normalize_explicit_time_indices_config(
        explicit_time_indices_by_dataset: dict[str, dict[str, list[int]]] | None,
    ) -> dict[str, dict[str, np.ndarray]]:
        normalized: dict[str, dict[str, np.ndarray]] = {}
        for dataset_name, dataset_cfg in (explicit_time_indices_by_dataset or {}).items():
            if not hasattr(dataset_cfg, "get"):
                raise ValueError(
                    f"Explicit time indices for dataset '{dataset_name}' must define `input` and `target`."
                )

            raw_input = dataset_cfg.get("input", None)
            raw_target = dataset_cfg.get("target", None)
            if raw_input is None or raw_target is None:
                raise ValueError(
                    f"Explicit time indices for dataset '{dataset_name}' must define both `input` and `target`."
                )

            input_indices = np.array(sorted({int(value) for value in raw_input}), dtype=np.int64)
            target_indices = np.array(sorted({int(value) for value in raw_target}), dtype=np.int64)
            if len(input_indices) == 0:
                raise ValueError(
                    f"Explicit time indices for dataset '{dataset_name}' require a non-empty `input`."
                )
            if np.any(input_indices < 0) or np.any(target_indices < 0):
                raise ValueError(
                    f"Explicit time indices for dataset '{dataset_name}' must be non-negative."
                )

            normalized[str(dataset_name)] = {
                "input": input_indices,
                "target": target_indices,
            }
        return normalized

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
        return self._collect("supporting_arrays")

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
        """Return valid date indices.

        A date t is valid if we can sample the elements t + i
        for every relative_date_index i across all datasets.

        Returns the intersection of valid indices from all datasets.
        """
        valid_date_indices_by_dataset: dict[str, np.ndarray] = {}
        for name, ds in self.datasets.items():
            relative_indices = self.data_relative_date_indices_by_dataset[name]
            valid_date_indices = get_usable_indices(
                ds.missing,
                len(ds.dates),
                relative_indices,
                ds.trajectory_ids if ds.has_trajectories else None,
            )
            valid_date_indices_by_dataset[name] = valid_date_indices

            if len(valid_date_indices) == 0:
                msg = f"No valid date indices found for dataset '{name}': \n{ds}"
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
        """
        self.global_rank = global_rank
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

        self.sample_comm_group_id = model_comm_group_id
        self.sample_comm_num_groups = model_comm_num_groups

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
        sanity_rnd = self.rng.random(1)
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
        for name, dataset in self.datasets.items():
            shard_start, shard_end = get_partition_range(self.shard_shapes[name], self.reader_group_rank)
            time_indices = self._resolve_dataset_time_indices(name, index)
            if log_timing:
                t_ds = time.perf_counter()
            x[name] = self._get_dataset_sample(
                dataset_name=name,
                dataset=dataset,
                time_indices=time_indices,
                grid_shard_indices=slice(shard_start, shard_end),
            )
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

    def _get_dataset_sample(
        self,
        *,
        dataset_name: str,
        dataset,
        time_indices: slice | int | list[int],
        grid_shard_indices: slice,
    ) -> torch.Tensor:
        mode = self._resolved_time_index_mode()
        anchor_dataset_name = getattr(self, "_anchor_dataset_name", None)
        use_sparse_aux = mode == "sparse" and anchor_dataset_name is not None and dataset_name != anchor_dataset_name
        if not use_sparse_aux:
            return dataset.get_sample(time_indices, grid_shard_indices)
        return self._get_sparse_dataset_sample(
            dataset_name=dataset_name,
            dataset=dataset,
            time_indices=time_indices,
            grid_shard_indices=grid_shard_indices,
        )

    def _get_sparse_dataset_sample(
        self,
        *,
        dataset_name: str,
        dataset,
        time_indices: slice | int | list[int],
        grid_shard_indices: slice,
    ) -> torch.Tensor:
        requested_indices = self._expand_time_indices(time_indices)
        valid_positions = [
            pos
            for pos, native_index in enumerate(requested_indices)
            if self._is_available_native_index(dataset, native_index)
        ]
        if len(valid_positions) == len(requested_indices):
            return dataset.get_sample(time_indices, grid_shard_indices)

        valid_indices = [requested_indices[pos] for pos in valid_positions]
        loaded = None
        if len(valid_indices) > 0:
            loaded = dataset.get_sample(valid_indices, grid_shard_indices)
            loaded = self._ensure_time_axis(loaded)

        if loaded is None:
            probe_idx = self._first_available_native_index(dataset)
            if probe_idx is None:
                raise ValueError(f"Dataset '{dataset_name}' has no available native indices for sparse loading.")
            probe = dataset.get_sample([probe_idx], grid_shard_indices)
            probe = self._ensure_time_axis(probe)
            output = torch.full(
                (len(requested_indices),) + tuple(probe.shape[1:]),
                torch.nan,
                dtype=torch.float32,
                device=probe.device,
            )
        else:
            dtype = loaded.dtype if loaded.is_floating_point() else torch.float32
            output = torch.full(
                (len(requested_indices),) + tuple(loaded.shape[1:]),
                torch.nan,
                dtype=dtype,
                device=loaded.device,
            )
            if not loaded.is_floating_point():
                loaded = loaded.float()

        if len(valid_positions) > 0:
            output[valid_positions] = loaded

        missing_count = len(requested_indices) - len(valid_positions)
        logged = self._sparse_missing_log_counts.get(dataset_name, 0)
        if logged < 5 and missing_count > 0:
            first_requested = requested_indices[0] if len(requested_indices) > 0 else -1
            LOGGER.info(
                "Sparse sample fill for dataset '%s': %d/%d requested native times unavailable at index=%d.",
                dataset_name,
                missing_count,
                len(requested_indices),
                first_requested,
            )
            self._sparse_missing_log_counts[dataset_name] = logged + 1

        return output

    @staticmethod
    def _expand_time_indices(time_indices: slice | int | list[int]) -> list[int]:
        if isinstance(time_indices, int):
            return [int(time_indices)]
        if isinstance(time_indices, slice):
            start = 0 if time_indices.start is None else int(time_indices.start)
            stop = int(time_indices.stop)
            step = 1 if time_indices.step is None else int(time_indices.step)
            return list(range(start, stop, step))
        return [int(v) for v in time_indices]

    @staticmethod
    def _is_available_native_index(dataset, native_index: int) -> bool:
        return 0 <= native_index < len(dataset.dates) and native_index not in dataset.missing

    @staticmethod
    def _first_available_native_index(dataset) -> int | None:
        if len(dataset.dates) == 0:
            return None
        missing = dataset.missing
        for idx in range(len(dataset.dates)):
            if idx not in missing:
                return idx
        return None

    @staticmethod
    def _ensure_time_axis(sample: torch.Tensor) -> torch.Tensor:
        if sample.ndim == 3:
            return sample.unsqueeze(0)
        return sample

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
            explicit_cfg = self.explicit_time_indices_by_dataset.get(name)
            if explicit_cfg is not None:
                input_model_indices = explicit_cfg["input"].astype(np.int64, copy=False)
                target_model_indices = explicit_cfg["target"].astype(np.int64, copy=False)
                model_indices = self._merge_input_and_required_indices(
                    input_model_indices=input_model_indices,
                    required_model_indices=target_model_indices,
                )
                rel_seconds = model_indices * self.timestep_seconds
                if np.any(rel_seconds % dataset_frequency_seconds != 0):
                    raise ValueError(
                        f"Dataset '{name}' explicit time indices {model_indices.tolist()} are not exact native "
                        f"timestamps for dataset frequency {ds.frequency}."
                    )
            else:
                exact_mask = model_relative_seconds % dataset_frequency_seconds == 0
                model_indices = self.model_relative_date_indices[exact_mask]
                input_model_indices = model_indices.astype(np.int64, copy=False)
                input_set = set(int(v) for v in input_model_indices.tolist())
                target_model_indices = np.array(
                    [int(v) for v in model_indices.tolist() if int(v) not in input_set],
                    dtype=np.int64,
                )

            if len(model_indices) == 0:
                msg = (
                    f"Dataset '{name}' has no exact native timestamps for requested model-relative "
                    f"indices {self.model_relative_date_indices.tolist()} with timestep {self.timestep}."
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

        self.input_model_relative_date_indices_by_dataset = input_model_indices_by_dataset
        self.target_model_relative_date_indices_by_dataset = target_model_indices_by_dataset
        return dataset_model_indices

    @staticmethod
    def _merge_input_and_required_indices(
        *,
        input_model_indices: np.ndarray,
        required_model_indices: np.ndarray,
    ) -> np.ndarray:
        """Merge and sort unique indices to keep deterministic time ordering."""
        return np.unique(
            np.concatenate([input_model_indices, required_model_indices]).astype(np.int64, copy=False),
        ).astype(np.int64, copy=False)

    def _to_native_relative_indices(self, dataset_name: str, model_relative_indices: np.ndarray) -> np.ndarray:
        dataset_frequency_seconds = frequency_to_seconds(self.datasets[dataset_name].frequency)
        rel_seconds = model_relative_indices * self.timestep_seconds
        if np.any(rel_seconds % dataset_frequency_seconds != 0):
            msg = f"Dataset '{dataset_name}' has non-exact native conversion for indices {model_relative_indices}."
            raise ValueError(msg)
        return (rel_seconds // dataset_frequency_seconds).astype(np.int64, copy=False)

    def _resolve_dataset_time_indices(self, dataset_name: str, index: int) -> slice | int | list[int]:
        mode = self._resolved_time_index_mode()
        anchor_dataset_name = getattr(self, "_anchor_dataset_name", None)
        if mode == "sparse" and anchor_dataset_name is not None and dataset_name != anchor_dataset_name:
            sparse_time_indices = self._resolve_sparse_dataset_time_indices(dataset_name=dataset_name, index=index)
            if sparse_time_indices is not None:
                return sparse_time_indices

        native_relative_indices = self.data_relative_date_indices_by_dataset[dataset_name]
        absolute_indices = index + native_relative_indices
        if len(absolute_indices) == 1:
            return int(absolute_indices[0])

        diffs = np.diff(absolute_indices)
        if len(diffs) > 0 and np.all(diffs == diffs[0]):
            step = int(diffs[0])
            start = int(absolute_indices[0])
            stop = int(absolute_indices[-1] + step)
            return slice(start, stop, step)

        return absolute_indices.tolist()

    def _build_dataset_date_index_maps(self) -> tuple[dict[str, np.ndarray | None], dict[str, dict[int, int] | None]]:
        dates_ns_by_dataset: dict[str, np.ndarray | None] = {}
        date_to_native_index_by_dataset: dict[str, dict[int, int] | None] = {}
        for dataset_name, dataset in self.datasets.items():
            dates_ns = self._dates_to_unix_ns(dataset.dates)
            if dates_ns is None:
                dates_ns_by_dataset[dataset_name] = None
                date_to_native_index_by_dataset[dataset_name] = None
                continue
            dates_ns = np.asarray(dates_ns, dtype=np.int64)
            dates_ns_by_dataset[dataset_name] = dates_ns
            date_to_native_index_by_dataset[dataset_name] = {int(date_ns): idx for idx, date_ns in enumerate(dates_ns)}
        return dates_ns_by_dataset, date_to_native_index_by_dataset

    @staticmethod
    def _dates_to_unix_ns(dates) -> np.ndarray | None:
        dates_array = np.asarray(dates)
        if np.issubdtype(dates_array.dtype, np.datetime64):
            return dates_array.astype("datetime64[ns]").astype(np.int64, copy=False)

        try:
            return np.array([np.datetime64(date_value, "ns").astype(np.int64) for date_value in dates], dtype=np.int64)
        except (TypeError, ValueError):
            return None

    def _resolve_sparse_dataset_time_indices(self, *, dataset_name: str, index: int) -> list[int] | int | None:
        anchor_dataset_name = getattr(self, "_anchor_dataset_name", None)
        if anchor_dataset_name is None:
            return None

        anchor_dates_ns = self._dates_ns_by_dataset.get(anchor_dataset_name)
        dataset_dates_ns = self._dates_ns_by_dataset.get(dataset_name)
        dataset_date_map = self._date_to_native_index_by_dataset.get(dataset_name)
        if anchor_dates_ns is None or dataset_dates_ns is None or dataset_date_map is None:
            return None

        if not (0 <= index < len(anchor_dates_ns)):
            return [-1] * len(self.model_relative_date_indices_by_dataset[dataset_name])

        anchor_date_ns = int(anchor_dates_ns[index])
        model_relative_indices = self.model_relative_date_indices_by_dataset[dataset_name]
        offsets_ns = model_relative_indices.astype(np.int64, copy=False) * self.timestep_seconds * 1_000_000_000
        requested_dates_ns = anchor_date_ns + offsets_ns

        resolved_native_indices = []
        for date_ns in requested_dates_ns:
            native_index = dataset_date_map.get(int(date_ns))
            if native_index is None:
                native_index = self._nearest_native_index(
                    dataset_name=dataset_name,
                    dataset_dates_ns=dataset_dates_ns,
                    target_date_ns=int(date_ns),
                )
            resolved_native_indices.append(-1 if native_index is None else int(native_index))

        if len(resolved_native_indices) == 1:
            return int(resolved_native_indices[0])
        return resolved_native_indices

    def _nearest_native_index(
        self,
        *,
        dataset_name: str,
        dataset_dates_ns: np.ndarray,
        target_date_ns: int,
    ) -> int | None:
        if len(dataset_dates_ns) == 0:
            return None

        insertion_index = int(np.searchsorted(dataset_dates_ns, target_date_ns, side="left"))
        candidate_indices = []
        if insertion_index < len(dataset_dates_ns):
            candidate_indices.append(insertion_index)
        if insertion_index > 0:
            candidate_indices.append(insertion_index - 1)
        if not candidate_indices:
            return None

        nearest_index = min(candidate_indices, key=lambda idx: abs(int(dataset_dates_ns[idx]) - target_date_ns))
        tolerance_ns = self._nearest_time_tolerance_ns(dataset_name)
        if abs(int(dataset_dates_ns[nearest_index]) - target_date_ns) > tolerance_ns:
            return None
        return nearest_index

    def _nearest_time_tolerance_ns(self, dataset_name: str) -> int:
        dataset_frequency_seconds = frequency_to_seconds(self.datasets[dataset_name].frequency)
        return max(1, (dataset_frequency_seconds * 1_000_000_000) // 2)

    @staticmethod
    def _format_time_indices_for_log(time_indices: slice | int | list[int]) -> str:
        if isinstance(time_indices, slice):
            return f"{time_indices.start}:{time_indices.stop}:{time_indices.step}"
        if isinstance(time_indices, int):
            return str(time_indices)
        if len(time_indices) <= 8:
            return str(time_indices)
        return f"{time_indices[:4]}...{time_indices[-2:]}(n={len(time_indices)})"
