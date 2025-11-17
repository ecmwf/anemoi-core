# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
import random
from collections.abc import Callable
from functools import cached_property

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import IterableDataset

from anemoi.training.data.grid_indices import BaseGridIndices
from anemoi.training.utils.seeding import get_base_seed
from anemoi.training.utils.usable_indices import get_usable_indices

LOGGER = logging.getLogger(__name__)


class NativeGridDataset(IterableDataset):
    """Iterable dataset for AnemoI data on the arbitrary grids."""

    def __init__(
        self,
        data_reader: Callable,
        grid_indices: type[BaseGridIndices],
        relative_date_indices: list,
        timestep: str = "6h",
        shuffle: bool = True,
        label: str = "generic",
        num_gpus_per_ens: int = 1,
        num_gpus_per_model: int = 1,
    ) -> None:
        """Initialize (part of) the dataset state.

        Parameters
        ----------
        data_reader : Callable
            user function that opens and returns the anemoi-datasets array data
        grid_indices : Type[BaseGridIndices]
            indices of the grid to keep. Defaults to None, which keeps all spatial indices.
        relative_date_indices: list
            list of time indices to load from the data relative to the current sample i in __iter__
        timestep : int, optional
            the time frequency of the samples, by default '6h'
        shuffle : bool, optional
            Shuffle batches, by default True
        label : str, optional
            label for the dataset, by default "generic"
        num_gpus_per_ens : int, optional
            Number of GPUs per ensemble, by default 1
        num_gpus_per_model : int, optional
            Number of GPUs per model, by default 1
        """
        self.label = label

        self.data = data_reader

        self.timestep = timestep
        self.grid_indices = grid_indices

        # lazy init
        self.n_samples_per_epoch_total: int = 0
        self.n_samples_per_epoch_per_worker: int = 0

        self.num_gpus_per_ens = num_gpus_per_ens
        self.num_gpus_per_model = num_gpus_per_model

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
        self.shuffle = shuffle

        # Data dimensions
        self.ensemble_dim: int = 2
        self.ensemble_size = self.data.shape[self.ensemble_dim]

        # relative index of dates to extract
        self.relative_date_indices = relative_date_indices

    @cached_property
    def statistics(self) -> dict:
        """Return dataset statistics."""
        return self.data.statistics

    @cached_property
    def statistics_tendencies(self) -> dict:
        """Return dataset tendency statistics."""
        try:
            return self.data.statistics_tendencies(self.timestep)
        except (KeyError, AttributeError):
            return None

    @cached_property
    def metadata(self) -> dict:
        """Return dataset metadata."""
        return self.data.metadata()

    @cached_property
    def supporting_arrays(self) -> dict:
        """Return dataset supporting_arrays."""
        return self.data.supporting_arrays()

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

        A date t is valid if we can sample the elements t + i
        for every relative_date_index i.
        """
        return get_usable_indices(
            self.data.missing,
            len(self.data),
            np.array(self.relative_date_indices, dtype=np.int64),
            self.data.trajectory_ids,
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
            "model_comm_group_rank %d, model_comm_num_groups %d, reader_group_rank %d",
            global_rank,
            model_comm_group_id,
            model_comm_group_rank,
            model_comm_num_groups,
            reader_group_rank,
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

        LOGGER.info(
            "NativeGridDataset.set_group_info(): global_rank %d, ens_comm_group_id %d, "
            "ens_comm_group_rank %d, ens_comm_num_groups %d, reader_group_rank %d",
            self.global_rank,
            ens_comm_group_id,
            ens_comm_group_rank,
            ens_comm_num_groups,
            self.reader_group_rank,
        )

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
        shard_size = len(self.valid_date_indices) // self.sample_comm_num_groups
        shard_start = self.sample_comm_group_id * shard_size
        shard_end = (self.sample_comm_group_id + 1) * shard_size

        shard_len = shard_end - shard_start
        self.n_samples_per_worker = shard_len // n_workers

        low = shard_start + worker_id * self.n_samples_per_worker
        high = min(shard_start + (worker_id + 1) * self.n_samples_per_worker, shard_end)
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
            (
                "Worker %d (%s, pid %d, glob. rank %d, model comm group %d, "
                "group_rank %d, seed group id %d, base_seed %d, sanity rnd %f)"
            ),
            worker_id,
            self.label,
            os.getpid(),
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
            self.sample_comm_group_id,
            base_seed,
            sanity_rnd,
        )

    def __iter__(self) -> torch.Tensor:
        """Return an iterator over the dataset.

        The datasets are retrieved by anemoi.datasets from anemoi datasets. This iterator yields
        chunked batches for DDP and sharded training.

        Currently it receives data with an ensemble dimension, which is discarded for
        now. (Until the code is "ensemble native".)
        """
        if self.shuffle:
            shuffled_chunk_indices = self.rng.choice(
                self.valid_date_indices,
                size=len(self.valid_date_indices),
                replace=False,
            )[self.chunk_index_range]
        else:
            shuffled_chunk_indices = self.valid_date_indices[self.chunk_index_range]

        LOGGER.debug(
            (
                "Worker pid %d, label %s, worker id %d, global_rank %d, "
                "model comm group %d, group_rank %d, seed comm group id %d, using indices[0:10]: %s"
            ),
            os.getpid(),
            self.label,
            self.worker_id,
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
            self.sample_comm_group_id,
            shuffled_chunk_indices[:10],
        )

        for i in shuffled_chunk_indices:
            start = i + self.relative_date_indices[0]
            end = i + self.relative_date_indices[-1] + 1
            timeincrement = self.relative_date_indices[1] - self.relative_date_indices[0]
            # NOTE: this is temporary until anemoi datasets allows indexing with arrays or lists
            # data[start...] will be replaced with data[self.relative_date_indices + i]

            grid_shard_indices = self.grid_indices.get_shard_indices(self.reader_group_rank)
            if isinstance(grid_shard_indices, slice):
                # Load only shards into CPU memory
                x = self.data[start:end:timeincrement, :, :, grid_shard_indices]

            else:
                # Load full grid in CPU memory, select grid_shard after
                # Note that anemoi-datasets currently doesn't support slicing + indexing
                # in the same operation.
                x = self.data[start:end:timeincrement, :, :, :]
                x = x[..., grid_shard_indices]  # select the grid shard
            x = rearrange(x, "dates variables ensemble gridpoints -> dates ensemble gridpoints variables")
            self.ensemble_dim = 1

            yield torch.from_numpy(x)

    def __repr__(self) -> str:
        return f"""
            {super().__repr__()}
            Dataset: {self.data}
            Relative dates: {self.relative_date_indices}
        """
