# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
import os
import random
from functools import cached_property
from typing import Callable

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info

from anemoi.training.utils.seeding import get_base_seed
from anemoi.training.utils.usable_indices import get_usable_indices

LOGGER = logging.getLogger(__name__)


class NativeGridDataset(IterableDataset):
    """Iterable dataset for AnemoI data on the arbitrary grids."""

    def __init__(
        self,
        data_reader: Callable,
        rollout: int = 1,
        multistep: int = 1,
        timeincrement: int = 1,
        model_comm_group_rank: int = 0,
        model_comm_group_id: int = 0,
        model_comm_num_groups: int = 1,
        shuffle: bool = True,
        label: str = "generic",
    ) -> None:
        """Initialize (part of) the dataset state.

        Parameters
        ----------
        data_reader : Callable
            user function that opens and returns the zarr array data
        rollout : int, optional
            length of rollout window, by default 12
        timeincrement : int, optional
            time increment between samples, by default 1
        multistep : int, optional
            collate (t-1, ... t - multistep) into the input state vector, by default 1
        model_comm_group_rank : int, optional
            process rank in the torch.distributed group (important when running on multiple GPUs), by default 0
        model_comm_group_id: int, optional
            device group ID, default 0
        model_comm_num_groups : int, optional
            total number of device groups, by default 1
        shuffle : bool, optional
            Shuffle batches, by default True
        label : str, optional
            label for the dataset, by default "generic"

        """
        self.label = label

        self.data = data_reader

        self.rollout = rollout
        self.timeincrement = timeincrement

        # lazy init
        self.n_samples_per_epoch_total: int = 0
        self.n_samples_per_epoch_per_worker: int = 0

        # DDP-relevant info
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.model_comm_group_id = model_comm_group_id
        self.global_rank = int(os.environ.get("SLURM_PROCID", "0"))

        # additional state vars (lazy init)
        self.n_samples_per_worker = 0
        self.chunk_index_range: np.ndarray | None = None
        self.shuffle = shuffle

        # Data dimensions
        self.multi_step = multistep
        assert self.multi_step > 0, "Multistep value must be greater than zero."
        self.ensemble_dim: int = 2
        self.ensemble_size = self.data.shape[self.ensemble_dim]

    @cached_property
    def statistics(self) -> dict:
        """Return dataset statistics."""
        return self.data.statistics

    @cached_property
    def metadata(self) -> dict:
        """Return dataset metadata."""
        return self.data.metadata()

    @cached_property
    def name_to_index(self) -> dict:
        """Return dataset statistics."""
        return self.data.name_to_index

    @cached_property
    def resolution(self) -> dict:
        """Return dataset resolution."""
        return self.data.resolution

    @cached_property
    def valid_date_indices(self) -> np.ndarray:
        """Return valid date indices.

        A date t is valid if we can sample the sequence
            (t - multistep + 1, ..., t + rollout)
        without missing data (if time_increment is 1).

        If there are no missing dates, total number of valid ICs is
        dataset length minus rollout minus additional multistep inputs
        (if time_increment is 1).
        """
        return get_usable_indices(self.data.missing, len(self.data), self.rollout, self.multi_step, self.timeincrement)

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        """Called by worker_init_func on each copy of dataset.

        This initialises after the worker process has been spawned.

        Parameters
        ----------
        n_workers : int
            Number of workers
        worker_id : int
            Worker ID

        """
        self.worker_id = worker_id

        # Divide this equally across shards (one shard per group!)
        shard_size = len(self.valid_date_indices) // self.model_comm_num_groups
        shard_start = self.model_comm_group_id * shard_size
        shard_end = (self.model_comm_group_id + 1) * shard_size

        shard_len = shard_end - shard_start
        self.n_samples_per_worker = shard_len // n_workers

        low = shard_start + worker_id * self.n_samples_per_worker
        high = min(shard_start + (worker_id + 1) * self.n_samples_per_worker, shard_end)

        LOGGER.debug(
            "Worker %d (pid %d, global_rank %d, model comm group %d) has low/high range %d / %d",
            worker_id,
            os.getpid(),
            self.global_rank,
            self.model_comm_group_id,
            low,
            high,
        )

        self.chunk_index_range = self.valid_date_indices[np.arange(low, high, dtype=np.uint32)]

        # each worker must have a different seed for its random number generator,
        # otherwise all the workers will output exactly the same data
        # should we check lightning env variable "PL_SEED_WORKERS" here?
        # but we alwyas want to seed these anyways ...

        base_seed = get_base_seed()

        seed = (
            base_seed * (self.model_comm_group_id + 1) - worker_id
        )  # note that test, validation etc. datasets get same seed
        torch.manual_seed(seed)
        random.seed(seed)
        self.rng = np.random.default_rng(seed=seed)
        sanity_rnd = self.rng.random(1)

        LOGGER.debug(
            (
                "Worker %d (%s, pid %d, glob. rank %d, model comm group %d, "
                "group_rank %d, base_seed %d) using seed %d, sanity rnd %f"
            ),
            worker_id,
            self.label,
            os.getpid(),
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
            base_seed,
            seed,
            sanity_rnd,
        )

    def __iter__(self) -> torch.Tensor:
        """Return an iterator over the dataset.

        The datasets are retrieved by Anemoi Datasets from zarr files. This iterator yields
        chunked batches for DDP and sharded training.

        Currently it receives data with an ensemble dimension, which is discarded for
        now. (Until the code is "ensemble native".)
        """
        if self.shuffle:
            shuffled_chunk_indices = self.rng.choice(
                self.chunk_index_range,
                size=self.n_samples_per_worker,
                replace=False,
            )
        else:
            shuffled_chunk_indices = self.chunk_index_range

        LOGGER.debug(
            (
                "Worker pid %d, label %s, worker id %d, global_rank %d, "
                "model comm group %d, group_rank %d using indices[0:10]: %s"
            ),
            os.getpid(),
            self.label,
            self.worker_id,
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
            shuffled_chunk_indices[:10],
        )

        for i in shuffled_chunk_indices:
            start = i - (self.multi_step - 1) * self.timeincrement
            end = i + (self.rollout + 1) * self.timeincrement

            x = self.data[start : end : self.timeincrement]
            x = rearrange(x, "dates variables ensemble gridpoints -> dates ensemble gridpoints variables")
            self.ensemble_dim = 1

            yield torch.from_numpy(x)

    def __repr__(self) -> str:
        return f"""
            {super().__repr__()}
            Dataset: {self.data}
            Rollout: {self.rollout}
            Multistep: {self.multi_step}
            Timeincrement: {self.timeincrement}
        """


def worker_init_func(worker_id: int) -> None:
    """Configures each dataset worker process.

    Calls WeatherBenchDataset.per_worker_init() on each dataset object.

    Parameters
    ----------
    worker_id : int
        Worker ID

    Raises
    ------
    RuntimeError
        If worker_info is None

    """
    worker_info = get_worker_info()  # information specific to each worker process
    if worker_info is None:
        LOGGER.error("worker_info is None! Set num_workers > 0 in your dataloader!")
        raise RuntimeError
    dataset_obj = worker_info.dataset  # the copy of the dataset held by this worker process.
    dataset_obj.per_worker_init(
        n_workers=worker_info.num_workers,
        worker_id=worker_id,
    )
