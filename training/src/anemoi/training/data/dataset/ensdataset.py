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
from collections.abc import Callable

import numpy as np
import torch
from einops import rearrange

from anemoi.training.data.grid_indices import BaseGridIndices
from anemoi.training.utils.seeding import get_base_seed

from .singledataset import NativeGridDataset

LOGGER = logging.getLogger(__name__)


class EnsNativeGridDataset(NativeGridDataset):
    """Iterable ensemble dataset for AnemoI data on the arbitrary grids."""

    def __init__(
        self,
        data_reader: Callable,
        grid_indices: type[BaseGridIndices],
        relative_date_indices: list,
        timestep: str = "6h",
        shuffle: bool = True,
        label: str = "generic",
        ens_members_per_device: int = 1,
        num_gpus_per_ens: int = 1,
        num_gpus_per_model: int = 1,
    ) -> None:
        """Initialize (part of) the dataset state.

        Parameters
        ----------
        data_reader : Callable
            user function that opens and returns the zarr array data
        relative_date_indices: list
            list of time indices to load from the data relative to the current sample i in __iter__
        ens_members_per_device: int, optional
            number of ensemble members input for each GPU device, by default 1
        timestep: str, optional
            timestep of the data, by default "6h"
        shuffle : bool, optional
            Shuffle batches, by default True
        ens_members_per_device : int, optional
            Number of ensemble members input for each GPU device, by default 1
        num_gpus_per_ens : int, optional
            Number of GPUs per ensemble, by default 1
        num_gpus_per_model : int, optional
            Number of GPUs per model, by default 1

        Raises
        ------
        RuntimeError
            Multistep value cannot be negative.
        """
        super().__init__(
            data_reader=data_reader,
            relative_date_indices=relative_date_indices,
            timestep=timestep,
            shuffle=shuffle,
            grid_indices=grid_indices,
            label=label,
        )

        # Lazy init
        self.ens_members_per_device = ens_members_per_device
        self.num_gpus_per_ens = num_gpus_per_ens
        self.num_gpus_per_model = num_gpus_per_model

        # lazy init model and reader group info, will be set by the DDPGroupStrategy:
        self.ens_comm_group_rank = 0
        self.ens_comm_num_groups = 1
        self.ens_comm_group_id = 0

    @property
    def num_eda_members(self) -> int:
        """Return number of EDA members."""
        return self.data.shape[2] - 1

    @property
    def eda_flag(self) -> bool:
        """Return whether EDA is enabled."""
        return self.data.shape[2] > 1

    def sample_eda_members(self, num_eda_members: int = 9) -> np.ndarray:
        """Subselect EDA ensemble members assigned to the current device."""
        tot_ens = self.ens_members_per_device * self.num_gpus_per_ens // self.num_gpus_per_model

        assert (
            tot_ens <= num_eda_members
        ), f"Can't generate an ensemble of size {tot_ens} from {num_eda_members} EDA perturbations"

        eda_member_gen_idx = self.rng_inicond_sampling.choice(range(num_eda_members), size=tot_ens, replace=False)
        offset = 1  # index=0 analysis, index=1 EDA recentred
        eda_member_gen_idx += offset

        effective_rank = self.ens_comm_group_rank // self.num_gpus_per_model
        eda_member_idx = np.sort(
            eda_member_gen_idx[
                effective_rank * self.ens_members_per_device : self.ens_members_per_device * (1 + effective_rank)
            ],
        )

        LOGGER.debug(
            "GPU with global rank %s, comm_group_id %s, comm_group_rank %s will receive EDA member(s) %s",
            self.global_rank,
            self.ens_comm_group_id,
            self.ens_comm_group_rank,
            eda_member_gen_idx,
        )

        return eda_member_gen_idx, eda_member_idx

    def set_comm_group_info(
        self,
        global_rank: int,
        model_comm_group_id: int,
        model_comm_group_rank: int,
        model_comm_num_groups: int,
        ens_comm_group_id: int,
        ens_comm_group_rank: int,
        ens_comm_num_groups: int,
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
        ens_comm_group_id : int
            Ensemble communication group ID
        ens_comm_group_rank : int
            Ensemble communication group rank
        ens_comm_num_groups : int
            Number of ensemble communication groups
        reader_group_rank : int
            Reader group rank
        reader_group_size : int
            Reader group size
        """
        self.global_rank = global_rank
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.ens_comm_group_id = ens_comm_group_id
        self.ens_comm_group_rank = ens_comm_group_rank
        self.ens_comm_num_groups = ens_comm_num_groups
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

        self.sample_comm_group_id = ens_comm_group_id  # groups that work on the same sample / batch
        self.sample_comm_num_groups = ens_comm_num_groups

        assert self.reader_group_size >= 1, "reader_group_size must be positive"

        LOGGER.info(
            "NativeGridDataset.set_group_info(): global_rank %d, model_comm_group_id %d, "
            "model_comm_group_rank %d, model_comm_num_groups %d, reader_group_rank %d",
            global_rank,
            model_comm_group_id,
            model_comm_group_rank,
            model_comm_num_groups,
            reader_group_rank,
        )

        LOGGER.info(
            "NativeGridDataset.set_group_info(): global_rank %d, ens_comm_group_id %d, "
            "ens_comm_group_rank %d, ens_comm_num_groups %d, reader_group_rank %d",
            global_rank,
            ens_comm_group_id,
            ens_comm_group_rank,
            ens_comm_num_groups,
            reader_group_rank,
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
        super().per_worker_init(n_workers, worker_id)

        base_seed = get_base_seed()
        seed = (
            base_seed * (self.sample_comm_group_id + 1) - worker_id
        )  # note that test, validation etc. datasets get same seed
        self.rng_inicond_sampling = np.random.default_rng(seed=seed)
        sanity_rnd = self.rng.random(1)
        sanity_rnd_ini = self.rng_inicond_sampling.random(1)

        LOGGER.info(
            (
                "Worker %d (%s, pid %d, glob. rank %d, model comm group %d, "
                "model comm group rank %d, ens comm group %d, ens comm group rank %d, "
                " seed group id %d, seed %d, sanity rnd %f, sanity rnd ini %f)"
            ),
            worker_id,
            self.label,
            os.getpid(),
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
            self.ens_comm_group_id,
            self.ens_comm_group_rank,
            self.sample_comm_group_id,
            seed,
            sanity_rnd,
            sanity_rnd_ini,
        )

    def __iter__(self):
        """Return an iterator over the dataset.

        The datasets are retrieved by Anemoi Datasets from zarr files. This iterator yields
        chunked batches for DDP and sharded training.
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
            # start and end time indices, for analysis and EDA
            start = i + self.relative_date_indices[0]
            end_an = i + self.relative_date_indices[-1] + 1
            timeincrement = self.relative_date_indices[1] - self.relative_date_indices[0]
            # NOTE: this is temporary until anemoi datasets allows indexing with arrays or lists
            # data[start...] will be replaced with data[self.relative_date_indices + i]
            end_eda = i + timeincrement

            if self.eda_flag:
                _, eda_member_idx = self.sample_eda_members(self.num_eda_members)
            else:
                eda_member_idx = None

            grid_shard_indices = self.grid_indices.get_shard_indices(self.reader_group_rank)
            if isinstance(grid_shard_indices, slice):
                # Load only shards into CPU memory
                x_an = self.data[start:end_an:timeincrement, :, 0:1, grid_shard_indices]
            else:
                # Load full grid in CPU memory, select grid_shard after
                # Note that anemoi-datasets currently doesn't support slicing + indexing
                # in the same operation.
                x_an = self.data[start:end_an:timeincrement, :, 0:1, ...]
                x_an = x_an[..., grid_shard_indices]  # select the grid shard
            x_an = rearrange(x_an, "dates variables ensemble gridpoints -> dates ensemble gridpoints variables")

            x_pert: torch.Tensor | None = None
            if self.eda_flag:
                if isinstance(grid_shard_indices, slice):
                    x_pert = self.data[start : end_eda : self.timeincrement, ..., grid_shard_indices]
                else:
                    x_pert = self.data[start : end_eda : self.timeincrement, ...]
                    x_pert = x_pert[..., grid_shard_indices]  # select the grid shard
                x_pert = rearrange(
                    x_pert[:, :, eda_member_idx, ...],
                    "dates variables ensemble gridpoints -> dates ensemble gridpoints variables",
                )

            if x_pert is not None:
                sample = (
                    torch.from_numpy(x_an),
                    torch.from_numpy(x_pert),
                )
            else:
                sample = (torch.from_numpy(x_an),)

            yield sample

    def __repr__(self) -> str:
        return (
            f"""
            {super().__repr__()}
            Dataset: {self.data}
            Relative dates: {self.relative_date_indices}
            EDA: {self.eda_flag}
            """
            f"""
            Number of EDA members:
            {self.num_eda_members}" if self.eda_flag else
            """
        )
